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
`--split`, `--sample-size 200000`, `--pipeline xgb --skip-tuning`:

| Split | Accuracy | F1-macro | F1-weighted | Stage 1 P / R / FPR (test) |
|---|---|---|---|---|
| `random` | 0.9944 | **0.9197** | 0.9945 | 0.251 / 0.979 / 0.593 |
| `temporal --temporal-block sub-label` *(default)* | 0.9612 | **0.7721** | 0.9597 | 0.250 / **0.804** / 0.491 |
| `temporal --temporal-block class` | 0.8807 | **0.2973** | 0.8483 | 0.679 / **0.393** / 0.038 |

> Las cifras de Stage 1 están medidas **sobre test**. Hasta la revisión del 8 de agosto de 2026 esta
> columna traía las de **validación** —el conjunto sobre el que la búsqueda F-beta *elige* el
> umbral—, que es un estadístico de ajuste y no una estimación. Ver
> [§ El punto de operación de Stage 1](#el-punto-de-operación-de-stage-1-medirlo-donde-se-elige-no-es-medirlo).

F1 por clase, con el número de herramientas de ataque que compone cada una:

| Clase | Herram. | `random` | `sub-label` | `class` |
|---|---|---|---|---|
| BENIGN | — | 1.00 | 0.98 | 0.94 |
| Botnet | 1 | 0.54 | **0.00** | **0.00** |
| BruteForce | 2 | 0.98 | 0.98 | **0.00** |
| DDoS | 1 | 0.99 | 0.98 | 0.63 |
| DoS | 5 | 0.98 | 0.84 | 0.44 |
| Scan | 1 | 0.99 | 0.80 | 0.01 |
| WebAttack | 3 | 0.96 | 0.83 | 0.06 |

Las tres filas son el **mismo dataset y la misma garantía cronológica**; lo único que cambia entre
las dos últimas es qué define un bloque temporal. Eso permite descomponer la caída en vez de
atribuirla entera a una sola causa:

```
F1-macro   random 0.9197  →  sub-label 0.7721  →  class 0.2973
           └──── 0.1476 ────┘└──────── 0.4748 ────────┘
              fuga (23,7 %)    artefacto taxonomía × tiempo (76,3 %)
```

**La fuga por casi-duplicados vale 0.148 de F1-macro.** Los otros 0.475 los produce el bloqueo por
clase canónica: `ATTACK_GROUPS` colapsa herramientas que corrieron **en serie** dentro del mismo
día, así que cortar por posición dentro de la clase holdea herramientas enteras (el test de
`tuesday/BruteForce` es 100 % SSH-Patator; el 35 % del test de `wednesday/DoS` es GoldenEye y
Heartbleed, ausentes de train). Es una pregunta legítima —generalizar a una herramienta no vista—
pero es otra, y solo se les plantea a las clases que tienen varias herramientas.

> Una versión anterior de este documento afirmaba que la caída a 0.2809 «es la magnitud de la
> fuga». Se equivocaba por un factor de ~4. El registro completo del error, la descomposición y el
> arreglo están en **[PARTICION-TEMPORAL.md](PARTICION-TEMPORAL.md)**.

El 0.9197 es el equivalente sin tuning del 0.9516 publicado — la fila `random` de esta tabla y la
tabla (A) miden lo mismo, así que la comparación honesta con el TFA es 0.9197 → 0.7721.

Mecanismo de la caída bajo `--temporal-block class`, en orden. Son síntomas correctos, todos aguas
abajo de la clave de bloqueo:

1. La composición de validación la fija la clave del bloque. Bajo `class`, la búsqueda F-beta elige
   un umbral de Stage 1 **mucho más conservador**: FPR cae de 0.491 a 0.038, pero el recall cae de
   0.804 a 0.393.
2. Stage 1 deja fuera el **61 % de los ataques**, y esos **nunca llegan a Stage 2**. Ningún acierto
   de Stage 2 puede recuperarlos.
3. Bajo `class`, **BruteForce ni siquiera llega a entrenar Stage 2**: el clasificador se entrena con
   **6 clases** sobre 26.678 filas. Bajo `sub-label` son **7 clases** sobre 93.142.

Ese umbral es único para todas las clases, así que el efecto no se queda en las clases
multi-herramienta: `Scan` y `DDoS` tienen una sola herramienta —su bloque se parte idénticamente
bajo ambas claves— y aun así caen a 0.01 y 0.63. En una cascada, contaminar la validación de tres
clases mueve el punto de operación de las siete.

**El punto débil real del diseño** es otro y sigue en pie: **Botnet cae 0.54 → 0.00 en los dos
protocolos temporales.** Tiene una sola sub-etiqueta y un solo día, así que su bloque se parte
idénticamente bajo ambas claves y no cabe confundirlo con una demanda de novedad: ese 0.54 era
memorización pura de ráfagas. Con ~1.950 flujos en el dataset completo, concentrados en un único
día, la muestra de 200k le deja 108 filas de entrenamiento y 24 de test: la clase no se aprende.
WebAttack, igual de minoritaria pero con tres herramientas, sí sobrevive al protocolo honesto
(0.83), lo que sitúa el problema de Botnet en el soporte y no en el protocolo.

### El punto de operación de Stage 1: medirlo donde se elige no es medirlo

Segundo error de concepto de esta rama, misma familia que el primero: **una métrica bien calculada,
presentada como si respondiera a otra pregunta.**

El umbral del Isolation Forest se elige maximizando F-beta sobre validación. `optimize_threshold`
dejaba anotado el punto de operación **de ese mismo conjunto**, y `evaluate()` lo copiaba dentro del
dict de métricas de test, junto a `accuracy` y `f1_macro`, que sí eran de test. Un lector del
`training_summary.json` —o de la tabla de arriba— leía `if_recall = 0.986` como una propiedad del
modelo sobre tráfico no visto. No lo era.

Medido con el umbral congelado sobre los dos conjuntos:

| Protocolo | S1 recall (validación) | S1 recall (test) | Δ |
|---|---|---|---|
| `random` | 0.978 | 0.979 | **+0.001** |
| `temporal` + `sub-label` | 0.986 | **0.804** | **−0.182** |
| `temporal` + `class` | 0.626 | **0.393** | **−0.233** |

**Bajo split aleatorio el hueco es cero.** Eso es lo que identifica la causa. Si el problema fuera
el sesgo de selección —el umbral es el argmax de F-beta sobre los puntos de operación de ese
conjunto— aparecería en las tres filas por igual, porque la búsqueda es idéntica. Solo aparece
cuando la partición es temporal, y la razón es estructural: bajo el protocolo temporal **validación
es el 15 % central de cada bloque y test el 15 % final**, así que validación está sistemáticamente
más cerca de train en el tiempo. Con split aleatorio validación y test son intercambiables y
sustituir uno por otro no cuesta nada; en cuanto dejan de serlo, la sustitución miente.

La consecuencia práctica es la peor posible para un IDS en cascada. Lo que Stage 1 no marca **nunca
llega a Stage 2** y sale como BENIGN: su recall es un techo duro para el sistema entero. La fracción
de ataques irrecuperables no era el 1,4 % publicado sino el **19,6 %** — un factor de 14.

Ninguna métrica end-to-end se mueve: `accuracy`, `f1_macro` y `f1_weighted` siempre salieron de
`predict(X_test)`. La descomposición 0.9197 → 0.7721 → 0.2973 se sostiene entera. Lo que cambia es
la columna de Stage 1 y la afirmación de cuántos ataques se pierden por el camino.

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
