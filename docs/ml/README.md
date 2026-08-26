# Documentación de ML

Sentinel detecta intrusiones sobre **dos modalidades de dato distintas**, en paralelo. No son dos
versiones del mismo modelo: son dos ontologías, dos taxonomías y dos rutas de inferencia separadas
a propósito.

| | Track de **eventos** | Track de **flujos** |
|---|---|---|
| Unidad de dato | Evento de log de infraestructura | Flujo de red bidireccional (CICFlowMeter) |
| Esquema | [`data/schemas/event.json`](../../data/schemas/event.json) | [`data/schemas/flow.json`](../../data/schemas/flow.json) |
| Features | 25 derivadas (ventanas, agregados por campaña) | 78 columnas CICFlowMeter |
| Taxonomía | `normal`, `brute_force`, `credential_stuffing`, `privilege_escalation`, `lateral_movement`, `data_exfiltration`, `reconnaissance` | `BENIGN`, `Botnet`, `BruteForce`, `DDoS`, `DoS`, `Scan`, `WebAttack` |
| Arquitectura | Ensemble: Isolation Forest ‖ XGBoost ‖ reglas → fusión | Cascada: Isolation Forest → XGBoost → fusión |
| Código | `ml/models/`, `ml/features/` | `ml/flow/` |
| Entrenamiento | `python -m ml.training.train` | `python -m ml.training.train_cicids` |
| Artefactos | `models/` | `models/flow/` |
| Endpoints | `POST /v1/predict`, `POST /v1/analyze/batch` | `POST /v1/predict/flow`, `POST /v1/analyze/flow/batch` |
| Datos de entrenamiento | Generador sintético (`data/generators/`) | CICIDS2017, 2,83 M de flujos reales |

Comparten [`ml/evaluation/metrics.py`](../../ml/evaluation/metrics.py) (agnóstico: numpy + sklearn) y
[`ml/evaluation/plots.py`](../../ml/evaluation/plots.py). **Nada más.** Cruzar las taxonomías produce
un modelo que predice `normal` para todo sin lanzar un solo error — ver
[CICIDS2017-TRACK.md § Por qué las taxonomías no se fusionan](CICIDS2017-TRACK.md#por-qué-las-taxonomías-no-se-fusionan).

## Índice

- **[CICIDS2017-TRACK.md](CICIDS2017-TRACK.md)** — el track de flujos en detalle: arquitectura de
  cascada, contrato de features, partición temporal, cómo entrenar dentro y fuera del cluster.
- **[memoria-cicids2017.md](memoria-cicids2017.md)** — memoria académica original del TFA (724
  líneas). Es el documento de origen; sus métricas son las de la columna "TFA" de más abajo y
  **están medidas con split aleatorio**.
- **[modules/](modules/)** — 12 fichas de módulo del TFA, aplanadas con prefijo de directorio
  (`pipelines_pipeline_a.md`, `models_xgboost_classifier.md`, …). Describen el código **antes** del
  port; los nombres de clase que citan ya no existen (ver la tabla de renombrados en
  CICIDS2017-TRACK.md).
- **[results/results.txt](results/results.txt)** — log completo del run de referencia del TFA
  (1 M de flujos, 2 h 3 min). Es la fuente de los números publicados.
- **[results/figures/](results/figures/)** — 15 figuras de ese run.
- **[presentacion/](presentacion/)** — presentación de defensa (23 diapositivas) y el script que
  la genera.

---

## Resultados

Hay **dos conjuntos de números** y confundirlos es el error más fácil de cometer con este
material. Se publican los dos con su procedencia explícita.

### A) Run de referencia del TFA — split **aleatorio**, con tuning

1.000.001 flujos muestreados, test de 150.001, `RandomizedSearchCV` 20×3 en Stage 2.
Fuente: [`results/results.txt`](results/results.txt).

| Modelo | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| **Cascade XGB** (Pipeline A) | 0.9936 | **0.9516** | 0.9936 |
| **Cascade MLP** (Pipeline B) | 0.9696 | **0.7508** | 0.9747 |

Stage 1 (Isolation Forest, umbral F-beta con β=2), idéntico en ambas cascadas por construcción:
**precision 0.2467 · recall 0.9682 · FPR 0.6008 · FNR 0.0318**. Son cifras de **validación** (el TFA
las tomaba del conjunto donde se elige el umbral), pero aquí eso no las invalida: con split
aleatorio validación y test son intercambiables y el hueco medido es de 0.001. Deja de serlo en
cuanto la partición es temporal — ver
[§ El punto de operación de Stage 1](#el-punto-de-operación-de-stage-1-medirlo-donde-se-elige-no-es-medirlo).

F1 por clase:

| Clase | Cascade XGB | Cascade MLP | Δ |
|---|---|---|---|
| BENIGN | 0.9962 | 0.9815 | −0.0146 |
| Botnet | 0.7616 | 0.1588 | −0.6027 |
| BruteForce | 0.9909 | 0.9578 | −0.0330 |
| DDoS | 0.9836 | 0.9819 | −0.0017 |
| DoS | 0.9762 | 0.9687 | −0.0074 |
| Scan | 0.9914 | 0.8523 | −0.1391 |
| WebAttack | 0.9615 | 0.3547 | −0.6068 |

> ⚠️ **Estos números son una cota optimista y la propia memoria lo reconoce (§10.1).** El split es
> un `train_test_split` estratificado **por fila** sobre una captura continua: flujos casi idénticos
> de la misma ráfaga de ataque caen simultáneamente en train y en test. `drop_duplicates()` quita
> las copias exactas, no los casi-duplicados. Sirven para comparar **XGBoost contra MLP** (ambos
> pagan el mismo sesgo); no sirven para estimar el rendimiento sobre tráfico no visto.

### B) Medido en este repo — split **temporal**, sin tuning

Ese sesgo es exactamente lo que corrige el track de flujos de sentinel. Mismo comando salvo el
`--split`, `--sample-size 200000`, `--pipeline xgb --skip-tuning`, medido el **22 de agosto de
2026** sobre los CSV reales:

| Split | Accuracy | F1-macro | F1-weighted | Stage 1 P / R / FPR (test) |
|---|---|---|---|---|
| `random` | 0.9944 | **0.9453** | 0.9945 | 0.250 / **0.976** / 0.596 |
| `temporal --temporal-block sub-label` *(default)* | 0.9194 | **0.6467** | 0.9052 | 0.335 / **0.560** / 0.226 |
| `temporal --temporal-block class` | 0.9165 | **0.6547** | 0.8999 | 0.329 / **0.573** / 0.237 |

Y el mismo default con el suelo de muestreo activado (`--class-sample-floor 2000`), que **no es la
línea base** porque mueve el punto de operación de Stage 1 — ver el párrafo de Botnet más abajo:

| Split | Accuracy | F1-macro | F1-weighted | Stage 1 P / R / FPR (test) |
|---|---|---|---|---|
| `random` | 0.9939 | 0.9837 | 0.9938 | 0.276 / 0.974 / 0.596 |
| `temporal --temporal-block sub-label` | 0.9525 | **0.7893** | 0.9462 | 0.293 / **0.819** / 0.463 |
| `temporal --temporal-block class` | 0.9438 | 0.7631 | 0.9362 | 0.294 / 0.817 / 0.459 |

> Las cifras de Stage 1 están medidas **sobre test**. Hasta la revisión del 8 de agosto de 2026 esta
> columna traía las de **validación** —el conjunto sobre el que la búsqueda F-beta *elige* el
> umbral—, que es un estadístico de ajuste y no una estimación. Ver
> [§ El punto de operación de Stage 1](#el-punto-de-operación-de-stage-1-medirlo-donde-se-elige-no-es-medirlo).

F1 por clase, con el número de herramientas de ataque que compone cada una:

| Clase | Herram. | `random` | `sub-label` | `class` | S1 recall bajo `sub-label` |
|---|---|---|---|---|---|
| BENIGN | — | 0.997 | 0.954 | 0.952 | 0.226 *(es FPR)* |
| Botnet | 1 | 0.708 | **0.000** | **0.000** | **0.000** |
| BruteForce | 2 | 0.968 | 0.934 | 0.969 | 0.973 |
| DDoS | 1 | 0.993 | 0.886 | 0.893 | 0.796 |
| DoS | 5 | 0.976 | 0.732 | 0.653 | 0.594 |
| Scan | 1 | 0.995 | **0.201** | **0.244** | **0.113** |
| WebAttack | 3 | 0.981 | 0.820 | 0.873 | 0.962 |

La última columna es la que hace legible el resto: **el F1 de Stage 2 solo se mide sobre las filas
que Stage 1 dejó pasar.** El 0.201 de `Scan` no dice que el clasificador no sepa nombrar un
PortScan; dice que Stage 1 le entrega 122 de sus 1.080 flujos de test.

Las tres filas son el **mismo dataset y la misma garantía cronológica**; lo único que cambia entre
las dos últimas es qué define un bloque temporal. Ese diseño se montó para poder descomponer la
caída en vez de atribuirla entera a una sola causa, y merece la pena ver qué ha dado la
descomposición en cada medición:

```
Agosto 2026, con order_idx derivado del orden alfabético de los ficheros
F1-macro   random 0.9197  →  sub-label 0.7721  →  class 0.2973
           └──── 0.1476 ────┘└──────── 0.4748 ────────┘
              fuga (23,7 %)    artefacto taxonomía × tiempo (76,3 %)

Re-medido con order_idx en orden de captura real
F1-macro   random 0.9453  →  sub-label 0.6467  →  class 0.6547
           └──── 0.2986 ────┘└──────── −0.0080 ────────┘
             protocolo         artefacto ya no medible
```

**La descomposición de agosto no sobrevive a la re-medición.** El artefacto de taxonomía nunca fue
un efecto directo sobre Stage 2: su vía era el umbral compartido de Stage 1, y con el orden de
captura corregido los dos protocolos aterrizan en umbrales casi iguales (recall de Stage 1 0.560 y
0.573), de modo que el efecto desaparece — `class` queda incluso 0.008 por encima. Lo que sí sigue
siendo cierto es el hecho del dataset: bajo `class`, el test de `tuesday/BruteForce` es 100 %
SSH-Patator y el 35 % del de `wednesday/DoS` es GoldenEye y Heartbleed, ausentes de train. Lo que ya
no se sostiene es que eso costara 0.475 de F1-macro; `BruteForce` puntúa hoy **mejor** bajo `class`
(0.969) que bajo `sub-label` (0.934).

Queda la diferencia entre el protocolo aleatorio y el temporal, **0.2986 de F1-macro**, y llamarla
«fuga» sin más sería repetir el error con otro número: la columna de recall de Stage 1 de la tabla
por clase muestra que buena parte se paga antes de que Stage 2 opine. Bajo split aleatorio Stage 1
marca ≥ 0.95 de cada clase de ataque; bajo el temporal marca 0.11 de `Scan` y 0.00 de `Botnet`.

> Dos versiones anteriores de este documento se equivocaron aquí, y las dos están registradas en
> **[PARTICION-TEMPORAL.md](PARTICION-TEMPORAL.md)**: la primera presentó la caída a 0.2809 como «la
> magnitud de la fuga» (error de factor ~4); la segunda midió esa descomposición con un `order_idx`
> que ordenaba los ficheros alfabéticamente en lugar de por hora de captura. El experimento de
> control —código actual, orden alfabético restaurado— reproduce exactamente las cifras de agosto,
> así que el movimiento tiene una sola causa y está aislada.

El 0.9453 es el equivalente sin tuning del 0.9516 publicado — la fila `random` de esta tabla y la
tabla (A) miden lo mismo, así que la comparación honesta con el TFA es 0.9453 → 0.6467.

El mecanismo que gobierna todas estas cifras es **el punto de operación único de Stage 1**. Es un
solo umbral para las siete clases, elegido maximizando F-beta sobre validación, y cualquier cosa que
cambie la composición de validación lo mueve: la clave del bloque (lo que se midió en agosto), el
orden de lectura de los ficheros, y el suelo de muestreo. Lo que Stage 1 no marca sale como BENIGN
sin que Stage 2 lo vea, así que su recall es el techo duro del sistema: bajo el default actual, el
**44,0 %** de los ataques es irrecuperable por construcción, y con el suelo activado el 18,1 %.

**El punto débil real del diseño** es otro y sigue en pie: **Botnet cae 0.71 → 0.00 en los dos
protocolos temporales.** Tiene una sola sub-etiqueta y un solo día, así que su bloque se parte
idénticamente bajo ambas claves y no cabe confundirlo con una demanda de novedad: ese 0.71 era
memorización pura de ráfagas.

Hasta agosto la explicación se detenía en el soporte —con la muestra de 200k la clase se queda en
108 filas de entrenamiento y 24 de test, y sobre 24 filas 0.00 y 0.50 son la misma medida— y esa
objeción ya no aplica, porque `--class-sample-floor 2000` la desactiva:

| Run | Botnet train | Botnet test | S1 recall en Botnet | **F1 de Stage 2 en Botnet** |
|---|---|---|---|---|
| `sub-label`, sin suelo | 108 | 24 | **0.000** | 0.000 |
| `sub-label`, suelo 2.000 | 1.363 | 293 | **1.000** | **0.000** |
| `random`, suelo 2.000 | 1.364 | 292 | 1.000 | 0.973 |

Con las 1.948 filas que la clase tiene, con Stage 1 entregándole el 100 % del test y con 12,6× más
datos de entrenamiento, **Stage 2 sigue en 0.00 bajo el protocolo temporal y en 0.97 bajo el
aleatorio, con el mismo muestreo y el mismo volumen**. Lo que falta no es soporte: es que un botnet capturado un solo
día con una sola herramienta, cortado cronológicamente, no deja nada transferible del primer 70 % al
último 15 %. El suelo no arregla Botnet; retira la única objeción metodológica al 0.00.

> **El suelo sube el titular (0.6467 → 0.7893) y no es por Botnet.** `Scan` pasa de 0.201 a 0.792,
> `DDoS` de 0.886 a 0.956 y `DoS` de 0.732 a 0.847 — **con menos filas de entrenamiento que antes**,
> porque el suelo se las quita para pagar las de las clases raras. Lo que las mejora es que las
> 4.897 filas de ataque añadidas (2,45 % de la muestra, restadas sobre todo a BENIGN) mueven el
> ratio BENIGN:ataque que ve la búsqueda F-beta, el umbral
> sube de 0.0655 a 0.0885 y el recall de Stage 1 de 0.560 a 0.819. Es un cambio del punto de
> operación disfrazado de opción de muestreo, y por eso viene **desactivado por defecto**.

### El punto de operación de Stage 1: medirlo donde se elige no es medirlo

Segundo error de concepto de esta rama, misma familia que el primero: **una métrica bien calculada,
presentada como si respondiera a otra pregunta.**

El umbral del Isolation Forest se elige maximizando F-beta sobre validación. `optimize_threshold`
dejaba anotado el punto de operación **de ese mismo conjunto**, y `evaluate()` lo copiaba dentro del
dict de métricas de test, junto a `accuracy` y `f1_macro`, que sí eran de test. Un lector del
`training_summary.json` —o de la tabla de arriba— leía `if_recall = 0.986` (la cifra de validación
de entonces) como una propiedad del modelo sobre tráfico no visto. No lo era.

Medido con el umbral congelado sobre los dos conjuntos:

| Protocolo | S1 recall (validación) | S1 recall (test) | Δ |
|---|---|---|---|
| `random` | 0.975 | 0.976 | **+0.001** |
| `temporal` + `sub-label` | 0.760 | **0.560** | **−0.200** |
| `temporal` + `class` | 0.770 | **0.573** | **−0.197** |

**Bajo split aleatorio el hueco es cero.** Eso es lo que identifica la causa. Si el problema fuera
el sesgo de selección —el umbral es el argmax de F-beta sobre los puntos de operación de ese
conjunto— aparecería en las tres filas por igual, porque la búsqueda es idéntica. Solo aparece
cuando la partición es temporal, y la razón es estructural: bajo el protocolo temporal **validación
es el 15 % central de cada bloque y test el 15 % final**, así que validación está sistemáticamente
más cerca de train en el tiempo. Con split aleatorio validación y test son intercambiables y
sustituir uno por otro no cuesta nada; en cuanto dejan de serlo, la sustitución miente.

La consecuencia práctica es la peor posible para un IDS en cascada. Lo que Stage 1 no marca **nunca
llega a Stage 2** y sale como BENIGN: su recall es un techo duro para el sistema entero. La fracción
de ataques irrecuperables no era el 1,4 % publicado sino el 19,6 % cuando esto se midió — y con el
orden de captura corregido, el **44,0 %**.

Ninguna métrica end-to-end se movió *por este arreglo*: `accuracy`, `f1_macro` y `f1_weighted`
siempre salieron de `predict(X_test)`. Lo que cambió fue la columna de Stage 1 y la afirmación de
cuántos ataques se pierden por el camino.

`training_summary.json` registra ahora los dos puntos, etiquetados y sin posibilidad de confundirlos:

```jsonc
"stage1_operating_point": {
  "cascade_xgb": {
    "selection": { "recall": 0.985991, ... },  // validación: de dónde salió el umbral
    "test":      { "recall": 0.803627, ... }   // held-out: lo que se reporta
  }
}
```

Y la lección, que es la misma que la de la partición: **un número honesto sobre el conjunto
equivocado sigue siendo un número deshonesto.** El primer error puso la frontera del split en el
sitio equivocado; este puso la frontera del *reporte* en el sitio equivocado. Los dos sobrevivieron
a una suite en verde porque ningún test puede saber qué pregunta creías estar haciendo.

### Cómo reproducir

```powershell
# (B), fila temporal por defecto (sub-label) — ~90 s con los CSV reales en api/csvs/
python -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 `
  --pipeline xgb --split temporal --skip-tuning --output-dir models/flow

# (B), fila temporal con bloqueo por clase — el protocolo duro, reportado aparte
python -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 `
  --pipeline xgb --split temporal --temporal-block class --skip-tuning `
  --output-dir "$env:TEMP\flow-coarse"

# (B), fila random — cuantifica la fuga
python -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 `
  --pipeline xgb --split random --skip-tuning --output-dir "$env:TEMP\flow-random"
```

Requiere **~8 GiB de RAM**: `load_cicids2017` concatena los ocho CSV antes de muestrear, así que
`--sample-size` no gobierna el pico de memoria (pico medido: 7,97 GiB). Los CSV no están en git
(844 MB); se descargan de [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), distribución
`MachineLearningCSV`.

---

## Nota sobre `results/figures/`

`pipeline_b_pr_tradeoff.png` y `pipeline_b_threshold_curve.png` son **byte a byte idénticos** a sus
homólogos de `pipeline_a`: ambas cascadas comparten Stage 1. No son dos resultados, son el mismo
resultado duplicado por el layout de salida del TFA. Lo mismo vale para el bloque
`stage1_operating_point` del `training_summary.json` de un run `--pipeline both`.
