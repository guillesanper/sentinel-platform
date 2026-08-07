# Track de flujos (CICIDS2017)

Detección de intrusiones sobre **flujos de red bidireccionales** con features CICFlowMeter, en
paralelo al track de eventos de log. Vive en [`ml/flow/`](../../ml/flow/) y llega hasta la API:
no es un experimento aislado.

El origen es un TFA sobre CICIDS2017 (ver [memoria-cicids2017.md](memoria-cicids2017.md)). El port
a sentinel no fue una copia: cambia la partición de datos, añade una ruta de serving que el TFA no
tenía, y renombra las clases para que no colisionen con las del track de eventos.

## Índice

- [Arquitectura de cascada](#arquitectura-de-cascada)
- [Por qué las taxonomías no se fusionan](#por-qué-las-taxonomías-no-se-fusionan)
- [El contrato de features](#el-contrato-de-features)
- [La partición temporal](#la-partición-temporal)
- [Cómo entrenar](#cómo-entrenar)
- [Formato en disco de una cascada](#formato-en-disco-de-una-cascada)
- [Inferencia](#inferencia)
- [Entrenamiento en cluster](#entrenamiento-en-cluster)
- [Trabajo futuro](#trabajo-futuro)
- [Mapa de renombrados desde el TFA](#mapa-de-renombrados-desde-el-tfa)

---

## Arquitectura de cascada

El track de eventos usa un **ensemble**: los tres detectores puntúan todos los eventos y la capa de
fusión promedia. El track de flujos usa una **cascada**: Stage 2 solo ve lo que Stage 1 marcó.

```
flujo (78 features CICFlowMeter)
        │
        ▼
┌───────────────────────────────┐
│ STAGE 1 — FlowAnomalyDetector │   Isolation Forest, no supervisado.
│ ml/flow/models/               │   Umbral optimizado por F-beta (β=2 por defecto):
│   isolation_forest.py         │   β>1 penaliza los ataques no detectados, porque
└───────────┬───────────────────┘   un falso negativo aquí es terminal.
            │
    ¿anómalo?├── NO ──►  BENIGN. Fin. Stage 2 nunca lo ve.
            │
           SÍ
            ▼
┌───────────────────────────────┐
│ STAGE 2 — FlowAttackClassifier│   XGBoost multiclase sobre las 7 clases.
│ ml/flow/models/               │   Devuelve una probabilidad por familia.
│   xgboost_classifier.py       │
└───────────┬───────────────────┘
            ▼
┌───────────────────────────────┐
│ FUSIÓN — FlowFusionLayer      │   threat_score = 0.3·anomalía
│ ml/flow/fusion.py             │                + 0.5·clasificación
└───────────┬───────────────────┘                + 0.2·reglas
            ▼
  {threat_score, severity, attack_type, confidence,
   is_anomaly, explanation[], recommended_actions[], attack_probabilities{}}
```

**La consecuencia de la cascada es que el recall de Stage 1 es un techo duro.** Lo que Stage 1 deja
pasar como benigno no tiene segunda oportunidad, por buena que sea la precisión de Stage 2. Con
partición temporal ese techo es el 62,6 % — ver [README.md § Resultados](README.md#resultados).
Por eso el umbral se optimiza con β=2 y no con F1.

Existe una segunda cascada, `CascadeMLPPipeline` (Stage 2 = red neuronal en NumPy pura: BatchNorm,
Adam, focal loss, backprop a mano, 761 líneas). Se conserva por valor académico y **está fuera de
la ruta de serving**: F1-macro 0.7508 contra 0.9516 de XGBoost sobre el mismo test. No se
re-exporta desde `ml.flow.pipelines`, no entra en la imagen de producción (`.dockerignore`) y
`train_cicids.py` la importa de forma perezosa.

### Capa de fusión

`threat_score` combina tres señales con pesos **0.3 / 0.5 / 0.2** (el mismo reparto que el track de
eventos, ver [`ml/models/fusion.py`](../../ml/models/fusion.py)):

- **Anomalía (0.3)** — score de Stage 1 normalizado.
- **Clasificación (0.5)** — probabilidad agregada de ataque de Stage 2.
- **Reglas (0.2)** — heurísticas de dominio sobre el flujo. Las del track de eventos (IP interna,
  horario laboral) no aplican aquí; las de flujos miran `Flow Duration`, `Flow Bytes/s`,
  `Destination Port` y conteos de flags.

`severity` sale de `FLOW_SEVERITY_THRESHOLDS`: `CRITICAL` ≥ 0.9, `HIGH` ≥ 0.7, `MEDIUM` ≥ 0.4,
`LOW` en el resto.

Dos propiedades del código que conviene no romper:

1. **La normalización del score de anomalía es una sigmoide** sobre `(raw_score − threshold) / 0.05`,
   monótona decreciente y exactamente 0.5 en el punto de operación. Es **presentación, no
   calibración**: hace legible un score de Isolation Forest cuyo rango depende del entrenamiento.
   **No es una probabilidad y no debe presentarse como tal.**
2. **`_classification_threat` es monótono en la probabilidad de ataque** en sus tres ramas. Es la
   propiedad que hereda `threat_score` y que fijan los tests. Si se toca la fórmula, hay que
   mantener la monotonía.

Cuando Stage 1 marca un flujo pero Stage 2 no sabe nombrarlo, `attack_type` es
**`unknown_anomaly`** — que no es una de las 7 clases. Es la respuesta honesta, no un `BENIGN`
por defecto.

---

## Por qué las taxonomías no se fusionan

| Track de eventos (`ml.models.classifier.ATTACK_TYPES`) | Track de flujos (`ml.flow.labels.CLASS_ORDER`) |
|---|---|
| `normal` | `BENIGN` |
| `brute_force` | `BruteForce` |
| `credential_stuffing` | `Botnet` |
| `privilege_escalation` | `DDoS` |
| `lateral_movement` | `DoS` |
| `data_exfiltration` | `Scan` |
| `reconnaissance` | `WebAttack` |

La tentación es evidente: `brute_force` ≈ `BruteForce`, ambas listas tienen 7 entradas. **Es una
coincidencia y unificarlas sería un error de modelado.** Las clases de eventos describen la
*intención* del actor sobre un log de autenticación; las de flujos describen el *patrón de tráfico*
que produce una herramienta. `DDoS` y `DoS` no tienen equivalente en un log de auth;
`privilege_escalation` no tiene firma de flujo. Los dos vectores de features son disjuntos (25
agregados de campaña contra 78 estadísticos de paquete) y ~40 tests están acoplados a índices de
feature del track de eventos.

El riesgo concreto no es la confusión conceptual, es este:
`AttackClassifier._encode_labels` mapea **etiqueta desconocida → `normal`, en silencio**. Meter
etiquetas CICIDS en el clasificador de eventos produce un modelo que predice `normal` para todo,
entrena sin lanzar una sola excepción y pasa cualquier smoke test que mire el exit code. Módulos y
clases separados hacen que ese envenenamiento sea imposible por construcción. (Se añadió además un
`logger.warning("unknown_label_mapped_to_normal", …)` para que deje rastro.)

Las 15 etiquetas crudas de CICIDS2017 se colapsan a las 7 canónicas en
[`ml/flow/labels.py`](../../ml/flow/labels.py) vía `ATTACK_GROUPS` (`DoS Hulk`, `DoS GoldenEye`,
`DoS slowloris`, `DoS Slowhttptest` y `Heartbleed` → `DoS`; `PortScan` e `Infiltration` → `Scan`;
`FTP-Patator` y `SSH-Patator` → `BruteForce`; `Bot` → `Botnet`; las tres `Web Attack …` →
`WebAttack`).

> **Trampa de encoding.** Las tres etiquetas `Web Attack` del CSV crudo contienen un **EN DASH**
> (U+2013) que llega mojibakeado como `ï¿½` por una lectura latin-1 previa. `map_label()` repara
> ambas formas. Si ese parche se pierde, las tres etiquetas dejan de reconocerse, sus filas se
> descartan y **la clase WebAttack desaparece del entrenamiento sin ningún error**. Lo protegen
> `tests/unit/test_flow_labels.py` (con un escape U+2013 explícito) y `conftest.py`
> (`CICIDS_MOJIBAKE_EN_DASH`, construido con `chr(cp)` para que un round-trip de encoding sobre el
> propio fichero de test rompa la suite en vez de pasar desapercibido).

---

## El contrato de features

**Este es el fallo más caro posible en este diseño, y por eso tiene tres capas de defensa.**

Un modelo entrenado sobre columnas en el orden `[Destination Port, Flow Duration, …]` y alimentado
en inferencia con `[Flow Duration, Destination Port, …]` **no lanza ninguna excepción**. Las formas
cuadran, los tipos cuadran, `predict()` devuelve un array perfectamente formado. Simplemente todas
las predicciones son basura. En una plataforma de seguridad eso es peor que caerse: un detector
silenciosamente roto genera confianza injustificada.

Las tres defensas:

1. **`FLOW_FEATURE_NAMES` es la única fuente de verdad del espacio de features.**
   `Final[tuple[str, ...]]` con las 78 columnas en orden canónico, en
   [`ml/flow/features.py`](../../ml/flow/features.py), derivadas de la cabecera real de los CSV.
   En entrenamiento, `loader.get_feature_columns()` es la única puerta a la matriz; nadie hace
   `[c for c in df.columns if c != "Label"]` a mano.

2. **`FlowFeatureExtractor` reordena por NOMBRE, nunca por orden de dict.** Un payload JSON con las
   claves permutadas produce exactamente el mismo vector. Features ausentes → `0.0`; `inf` y `NaN`
   → `0.0` (el mismo saneado que `clean_dataset` aplica en entrenamiento); salida `float32`.

3. **`CascadeInferenceService.load()` valida la lista persistida contra `FLOW_FEATURE_NAMES` y
   lanza `FeatureContractError` si difiere en longitud, contenido U ORDEN.** El caso del orden es el
   peligroso: mismos nombres, distinta posición, ninguna excepción natural. Tras un load rechazado
   `is_loaded` queda en `False` y `predict()` se niega a responder. **Esta validación no debe
   relajarse para que un test sea más cómodo.**

Simétricamente, `BaseCascadePipeline.save()` **se niega a escribir** (`RuntimeError`) si no tiene
`feature_names` que persistir, y `load()` exige `feature_names.json` (`FileNotFoundError`) y valida
que `n_features` cuadre con la longitud de la lista (`ValueError`).

> **`Fwd Header Length` aparece dos veces, y es correcto.** El CSV crudo trae la columna duplicada
> —un artefacto de CICFlowMeter documentado en la sección de EDA de
> [memoria-cicids2017.md](memoria-cicids2017.md), donde el mapa de correlación muestra su práctica
> identidad— y pandas le añade el sufijo `.1` al leer. `FLOW_FEATURE_NAMES` es **la vista de pandas**, así que
> contiene `Fwd Header Length` **y** `Fwd Header Length.1`. **No se pueden colapsar**: son dos
> posiciones reales de la matriz y el test de "sin duplicados" debe seguir pasando. Para clientes
> JSON que no toleran el punto existe el alias `"Fwd Header Length 1"` en `FLOW_FEATURE_ALIASES`.

### Riesgo hermano: `capture_day` y `order_idx` como features

La partición temporal (siguiente sección) añade dos columnas de procedencia al DataFrame. Si
alguna se cuela en la matriz de entrenamiento, el modelo aprende a **"detectar el viernes"** y
devuelve métricas espectaculares con cero valor operativo. `get_feature_columns()` las excluye
explícitamente junto a `Label`, y hay un test dedicado a ello. Era el riesgo nº 1 del diseño.

---

## La partición temporal

### El problema

El TFA usa un doble `train_test_split` estratificado **por fila**. Sobre una captura continua eso
es fuga de información: flujos casi idénticos de la misma ráfaga de ataque —mismo host, mismo
escenario, variaciones mínimas de timing— caen a la vez en train y en test. `drop_duplicates()`
elimina copias exactas, **no casi-duplicados**. La memoria lo admite en §10.1: las métricas de §8
son *"una cota optimista"*.

### Las restricciones reales

- Los CSV tienen **79 columnas y NO incluyen `Timestamp`**: es la distribución `MachineLearningCVE`.
  El único portador de tiempo es el **nombre del fichero**
  (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`), y `pd.concat` lo destruía.
- **Las filas sí conservan orden cronológico.** Medido por deciles sobre ese mismo fichero (225.745
  filas): decil 0 → 100 % BENIGN, deciles 1–8 → DDoS, decil 9 → 100 % BENIGN. Es la ventana del
  ataque, en orden.

Eso invalida las dos soluciones obvias:

| Opción ingenua | Por qué falla |
|---|---|
| Corte posicional global (primer 70 % train, último 15 % test) | Sobre el fichero de arriba, **test sería 100 % BENIGN**. Métricas sin sentido. |
| Holdout por día ("reservar los viernes", memoria §10.1) | Cada clase vive en **un solo día** (Mar=BruteForce, Mié=DoS, Jue-AM=WebAttack, Vie=Botnet/Scan/DDoS). Reservar el viernes deja DDoS, Scan y Botnet con **cero muestras de entrenamiento**. |

### El diseño adoptado: bloque (día × clase)

Dentro de cada grupo `(capture_day, class)`, ordenar por posición original y cortar
**70 / 15 / 15 cronológicamente**: los flujos más antiguos a train, los intermedios a val, los más
recientes a test.

- **Sin fuga futuro→pasado**: ningún flujo de test precede temporalmente a uno de train dentro de
  su escenario.
- **Cobertura de clase preservada** en los tres splits, a diferencia del holdout por día.
- Los casi-duplicados de una ráfaga son **contiguos**, así que caen mayoritariamente en el mismo
  bloque en vez de repartirse. Ahí es donde se recupera la honestidad de la métrica.

Implementación en [`ml/flow/loader.py`](../../ml/flow/loader.py):

- `load_cicids2017()` añade **`capture_day`** (del nombre de fichero, vía el mapa explícito
  `CAPTURE_DAYS`) y **`order_idx`**. Ambas sobreviven a `clean_dataset`,
  `filter_minority_classes` y `stratified_sample`.
- `split_dataset_temporal()` es el **default**. `split_dataset()` (aleatoria) se conserva para
  poder **cuantificar la fuga**, que es un resultado en sí mismo.
  `split_dataset_day_holdout()` implementa el protocolo secundario, que loguea explícitamente qué
  clases quedan fuera del train; se reporta aparte, nunca como métrica principal.

Cuatro detalles del comportamiento que sorprenden si no se conocen:

1. **`order_idx` es un contador global**, no una posición por fichero.
2. **Los splits temporales barajan cada parte.** La garantía cronológica la llevan los **valores**
   de `order_idx`, no el orden de las filas en el DataFrame resultante.
3. **Los bloques con menos de 3 filas van enteros a train**, con warning
   `temporal_block_not_splittable`.
4. **`clean_dataset` deduplica sobre `features + Label`**, nunca sobre todas las columnas — si
   incluyera `order_idx`, no habría duplicado posible.

El resultado de aplicar esto está en [README.md § Resultados](README.md#resultados): F1-macro cae de
0.9166 a 0.2809. **Esa caída es el entregable, no un fallo.**

---

## Cómo entrenar

```powershell
python -m ml.training.train_cicids --data-dir api/csvs --pipeline xgb
```

| Opción | Default | Notas |
|---|---|---|
| `--data-dir PATH` | `data` | **El default casi nunca es el correcto.** Los CSV están en `api/csvs/` y no se versionan (844 MB). |
| `--output-dir PATH` | `models/flow` | Raíz de cascadas, figuras y summary. |
| `--sample-size INT` | `1000000` | Muestreo estratificado; `0` usa el dataset completo. **No gobierna la memoria** (ver aviso abajo). |
| `--pipeline [xgb\|mlp\|both]` | `xgb` | `both` añade la comparación cruzada. |
| `--split [temporal\|random\|day-holdout]` | `temporal` | `random` es el split con fuga del TFA, conservado para cuantificarla. |
| `--test-days TEXT` | `friday` | Repetible. **Solo aplica con `day-holdout`**; ignorada en el resto. |
| `--beta FLOAT` | `2.0` | Peso F-beta del umbral de Stage 1. `>1` penaliza los ataques no detectados. |
| `--skip-tuning / --tune` | `--tune` | `--tune` lanza una `RandomizedSearchCV` 20×3 en Stage 2. Los smokes deben usar `--skip-tuning`. |
| `--plots / --no-plots` | `--no-plots` | Requiere matplotlib + seaborn (`requirements-dev.txt`, no en la imagen). |
| `--eda / --no-eda` | `--no-eda` | Figuras exploratorias. Misma dependencia. |
| `--seed INT` | `42` | Semilla global (numpy, random, sklearn, xgboost). |
| `--mlflow / --no-mlflow` | `--no-mlflow` | **NO-OP documentado**, ver abajo. |

Los defaults de las opciones se leen de una instancia de `FlowConfig`, no se repiten como
literales, para que `--help` y la dataclass no puedan divergir.

> ⚠️ **`--sample-size` no ahorra RAM.** `load_cicids2017` lee y concatena los ocho CSV **antes** de
> que `stratified_sample` recorte, así que el pico se paga siempre: **7,97 GiB residentes**
> medidos con `--sample-size 200000`. Los 844 MB en disco son ~8 GiB de columnas float64 más las
> copias de `pd.concat` y `clean_dataset`. Bajar `--sample-size` esperando ahorrar memoria no
> ahorra nada.

> **`--mlflow` es un NO-OP a propósito.** MLflow en el track de flujos está fuera del alcance del
> plan. La opción existe por simetría con `python -m ml.training.train`, avisa por stdout y por log
> (`flow_training_mlflow_not_implemented`), y el summary registra
> `"mlflow": {"requested": …, "logged": false}` en vez de dejarlo ambiguo. **Si algún día se
> implementa, ese campo es el contrato a respetar** — no debe degradarse a un booleano suelto.

### Salidas

`--output-dir` queda así (`--pipeline both --plots --eda`):

```
models/flow/
├── training_summary.json     estrategia de split, métricas, punto de operación, bloque mlflow
├── split_distribution.png    la partición temporal por día y clase — hace auditable el punto crítico
├── cascade_xgb/              artefactos + confusion_matrix.png, feature_importance.png,
│                             threshold_curve.png, pr_tradeoff.png
├── cascade_mlp/              ídem
├── comparison/               figuras de comparación cruzada
└── eda/                      figuras exploratorias
```

Dos comportamientos deliberados: los artefactos se guardan **directamente** en
`cascade_xgb/` (no en un subdirectorio `model/` como hacía el TFA), porque es lo que lee
`CascadeInferenceService.load()`; y **`_run_cascade` guarda antes de dibujar**, con las llamadas a
figuras envueltas en `try/except`. Un fallo de matplotlib no puede tirar un run que ya produjo un
artefacto servible.

`summary["split"]["test_days"]` es `null` salvo con `day-holdout`: rellenarlo siempre haría leer el
default `("friday",)` como si se hubiera reservado un día.

---

## Formato en disco de una cascada

Este es el contrato que consume [`ml/flow/serving.py`](../../ml/flow/serving.py):

```
models/flow/cascade_xgb/
├── feature_names.json          {format_version, pipeline, n_features, feature_names[]}
├── isolation_forest/
│   ├── isolation_forest.joblib
│   ├── if_threshold.npy
│   └── if_threshold_info.json  (opcional: ausente si nunca se optimizó el umbral)
└── xgboost/
    ├── xgboost.joblib
    └── xgb_label_encoder.joblib
```

`cascade_mlp/` es igual pero con `neural_network/` (`nn_weights.npz`, `nn_bn_running.npz`,
`nn_label_encoder.joblib`, `nn_scaler.joblib`) en lugar de `xgboost/`.

`feature_names.json` lleva `format_version` (`"1.0.0"`) para poder evolucionar el formato sin
romper artefactos en disco.

---

## Inferencia

### Forma de cable

```json
{
  "features": {"Destination Port": 80, "Flow Duration": 1532000, "...": 0},
  "src_ip": "192.168.10.5",
  "dst_ip": "10.0.1.20",
  "dst_port": 80,
  "protocol": "TCP"
}
```

Los metadatos viven **fuera** de `features`. La separación modelo/explicación es **estructural, no
una denylist**: nada que esté fuera de `features` puede llegar al vector, así que añadir un campo
de contexto nuevo no puede contaminar el modelo por olvido. `src_ip`, `dst_ip`, `dst_port` y
`protocol` alimentan la explicación y el log, nunca la predicción. (`feature_mapping()` también
acepta un dict plano de features, sin envolver.)

Esquema formal en [`data/schemas/flow.json`](../../data/schemas/flow.json) (JSON Schema draft
2020-12) y validación Pydantic en
[`api/models/flow_schemas.py`](../../api/models/flow_schemas.py).

**Las 78 features son individualmente opcionales** (ausente → 0.0), pero un payload cuyas claves no
incluyan **ningún** nombre reconocido se rechaza con **422**. Sin ese guard, un cliente con el
esquema equivocado recibiría un vector de ceros puntuado como si fuera un flujo real.

### Endpoints

| Endpoint | Método | Descripción | Auth | Rate limit |
|---|---|---|---|---|
| `/v1/predict/flow` | POST | Un flujo | JWT | 100/min |
| `/v1/analyze/flow/batch` | POST | Hasta 100 flujos | JWT | 10/min |

El batch **no** hace un bucle de predicciones individuales como el de eventos: pasa la matriz
entera a `predict_batch`, así que Stage 1 puntúa los 100 flujos en una pasada y Stage 2 recibe una
única llamada con los enrutados. Sobre cien flujos son dos invocaciones de modelo, no doscientas.

### Degradación

Tres decisiones que hay que conocer antes de tocar el arranque de la API:

1. **`/ready` lo decide SOLO el track de eventos.** Los checks `flow_detector` y `flow_classifier`
   se reportan pero **no cambian el código de estado**. Si un modelo de flujo ausente tumbase la
   probe, k8s sacaría el pod de servicio y se llevaría por delante los endpoints de eventos por un
   track opcional.
2. **Sin cascada cargada, los endpoints de flujo devuelven 503 y los de eventos siguen en 200.**
   Hay test explícito de esa degradación parcial.
3. **La `lifespan` captura `FileNotFoundError` y `FeatureContractError`.** Un artefacto obsoleto se
   loguea a nivel `error` pidiendo reentreno, pero **no aborta el arranque**.

### Métricas

`PREDICTION_SCORE` y `ATTACK_TYPE_COUNT` llevan una etiqueta **`track`** (`event` | `flow`); no hay
colectores duplicados. Las 7 queries de [`observability/`](../../observability/) envuelven estas
métricas en `sum(...)`, que agrega la etiqueta nueva, así que siguen funcionando sin cambios.
**Si se añade un track más: etiquetar, no duplicar.**

---

## Entrenamiento en cluster

[`k8s/base/ml/`](../../k8s/base/ml/) tiene dos CronJob y dos PVC:

| Objeto | Qué hace |
|---|---|
| `sentinel-training` (CronJob) | Track de eventos: `python -m ml.training.train … --threshold-strategy fbeta --beta 2.0 --no-mlflow` |
| `sentinel-flow-training` (CronJob) | Track de flujos: `python -m ml.training.train_cicids --data-dir /data/cicids …`. **`suspend: true`.** |
| `sentinel-models` (PVC) | RWO 5Gi, montado en `/app/models`. Sin él, `readOnlyRootFilesystem: true` + `emptyDir` perdía los modelos entrenados. |
| `sentinel-cicids-data` (PVC) | RWO 2Gi, read-only en `/data/cicids`. **Nace vacío.** |

Tres cosas que parecen defectos y no lo son:

- **`sentinel-flow-training` sale con `suspend: true`** porque su PVC de datos nace vacío: los
  844 MB de CSV están excluidos de la imagen por `.dockerignore`. Un job que lee un directorio
  vacío es la misma clase de defecto que uno que invoca un módulo inexistente. La receta de carga
  (`kubectl cp` desde `api/csvs/`) y el patch para reactivarlo están en la cabecera de
  [`cicids-data-pvc.yaml`](../../k8s/base/ml/cicids-data-pvc.yaml).
- **`--pipeline xgb --no-plots --no-eda` en ese CronJob no son preferencias, son restricciones de
  la imagen.** `.dockerignore` excluye `cascade_mlp.py`, `neural_network.py` y `plots.py`, y
  matplotlib/seaborn solo están en `requirements-dev.txt`. `--pipeline mlp` o `both` fallarían en
  el import. Están escritas explícitamente aunque coincidan con los defaults del CLI para que la
  restricción se lea en el manifiesto en vez de descubrirse en un traceback.
- **Ambos CronJob llevan `--no-mlflow`.** `MLFLOW_TRACKING_URI` apunta a un Service que este repo
  **no despliega**, y `log_to_mlflow` no tiene `try/except`: con `--mlflow` el job de eventos
  reventaría cada domingo *después* de haber guardado los modelos. La env var se conserva porque la
  lee la librería directamente, así que habilitarlo el día que exista el servidor es quitar una
  línea.

> **Precondición de serving, no resuelta.** El Deployment de la API **no monta ningún volumen de
> modelos**, ni de eventos ni de flujos, y **no puede montar `sentinel-models`**: los PVC son
> namespaced, ese vive en `ml` y la API en `platform`. Servir estos artefactos en cluster exige un
> paso de traspaso — object store (`MODELS_BUCKET` existe para eso y no está cableado en ningún
> extremo), un volumen RWX, o hornear los artefactos en la imagen. **No es una regresión del track
> de flujos: el de eventos siempre estuvo así.** Queda documentado como precondición en la cabecera
> de [`training-job.yaml`](../../k8s/base/ml/training-job.yaml).

---

## Trabajo futuro

**Migrar a `GeneratedLabelledFlows`.** CICIDS2017 se distribuye en dos formas. Este track usa
`MachineLearningCSV` (79 columnas, sin tiempo). La otra, `GeneratedLabelledFlows`, trae **85
columnas** con `Timestamp` real, `Flow ID`, IPs y puertos de origen y destino. Eso permitiría:

- Split por **marca de tiempo absoluta** en lugar de por bloque `(día, clase)` — sin depender de
  que las filas conserven orden dentro del fichero.
- **Agrupación por host**, que es la fuga que la partición actual no cierra: un atacante que
  aparece en train y en test sigue siendo el mismo atacante aunque sus flujos estén separados en
  el tiempo.
- Validar el mapa `CAPTURE_DAYS` contra el timestamp real en vez de contra el nombre de fichero.

Es la solución definitiva al problema que la partición temporal solo mitiga. Está **fuera del
alcance** de esta integración, no descartada.

Otras mejoras identificadas y no ejecutadas: MLflow en `train_cicids.py`; el **VALIDATION GATE**
que `sentinel_doc.md` §5.6 documenta y nadie aplica; y tests exhaustivos de `neural_network`
(gradient checking) — el smoke cubre el grueso del valor.

---

## Mapa de renombrados desde el TFA

`docs/ml/modules/` describe el código **antes** del port y usa los nombres viejos.

| TFA | sentinel |
|---|---|
| `IsolationForestModel` | `FlowAnomalyDetector` |
| `XGBoostClassifier` | `FlowAttackClassifier` |
| `PipelineA` | `CascadeXGBPipeline` |
| `PipelineB` | `CascadeMLPPipeline` |
| `BasePipeline` | `BaseCascadePipeline` |
| `Config` + singleton `cfg` | `FlowConfig` (explícito, sin singleton) |
| `compare_pipelines`, `pipeline_a/b` | `compare_models`, `model_a/b` |
| `main.py` (argparse) | `ml/training/train_cicids.py` (click) |
| `utils/logger.py` | structlog (eventos snake_case: `flow_training_*`, `cascade_*`, …) |

Los renombrados evitan colisión con `AnomalyDetector` / `AttackClassifier` del track de eventos y
con `ml/features/pipeline.py`, donde "pipeline" ya significa otra cosa.

### Importar del track de flujos

`ml/flow/__init__.py` **no re-exporta** `loader`, `models`, `pipelines`, `features`, `fusion` ni
`serving`, a propósito: `import ml.flow` sigue siendo barato. Hay que importar siempre el módulo
concreto:

```python
from ml.flow.serving import CascadeInferenceService   # sí
from ml.flow import serving                           # también
import ml.flow; ml.flow.serving...                    # no: no está enganchado
```

Y una restricción dura: **`api/Dockerfile` copia `ml/` entero instalando solo
`requirements.txt`**, donde no hay matplotlib. Ningún `__init__.py` de `ml/` puede importar
`plots.py`. Hay cinco test guards por subproceso que lo verifican.
