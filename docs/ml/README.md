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
**precision 0.2467 · recall 0.9682 · FPR 0.6008 · FNR 0.0318**.

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

| Split | Accuracy | F1-macro | F1-weighted | Stage 1 P / R / FPR |
|---|---|---|---|---|
| `random` | 0.9934 | **0.9166** | 0.9935 | 0.241 / 0.976 / 0.625 |
| `temporal` *(default)* | 0.8783 | **0.2809** | 0.8439 | 0.790 / 0.626 / 0.034 |

F1 por clase del run temporal:

| BENIGN | DDoS | DoS | Botnet | BruteForce | Scan | WebAttack |
|---|---|---|---|---|---|---|
| 0.94 | 0.62 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 |

**La caída de 0.9166 a 0.2809 en F1-macro no es una regresión: es la magnitud de la fuga.** El
0.9166 es el equivalente sin tuning del 0.9516 publicado — la fila `random` de esta tabla y la
tabla (A) miden lo mismo, así que la comparación honesta es 0.9166 → 0.2809, no 0.9516 → 0.2809.

Mecanismo de la caída, en orden:

1. Con validación temporal, la búsqueda F-beta elige un umbral de Stage 1 **mucho más
   conservador**: FPR cae de 0.625 a 0.034, pero el recall cae de 0.976 a 0.626.
2. Stage 1 deja fuera el **37 % de los ataques**, y esos **nunca llegan a Stage 2**. Ningún acierto
   de Stage 2 puede recuperarlos.
3. En el run temporal **BruteForce ni siquiera llega a entrenar Stage 2**: el clasificador guardado
   tiene **6 clases** (`BENIGN, Botnet, DDoS, DoS, Scan, WebAttack`), no 7.

Las cuatro clases a 0.00 son las minoritarias (Botnet ~2.000 filas, WebAttack ~850, ambas
concentradas en un único día) más Scan. **Ese es el punto débil real del diseño**, y con split
aleatorio quedaba enmascarado: Botnet marcaba 0.7616 y WebAttack 0.9615 memorizando ráfagas.

### Cómo reproducir

```powershell
# (B), fila temporal — ~90 s con los CSV reales en api/csvs/
python -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 `
  --pipeline xgb --split temporal --skip-tuning --output-dir models/flow

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
