# Módulo `config`

## Descripción General

El módulo `config` centraliza la totalidad de los parámetros configurables del sistema de detección y clasificación de ataques de red basado en CICIDS2017. Su propósito es eliminar valores *hardcoded* dispersos en el código fuente, garantizar la reproducibilidad experimental mediante la definición explícita de semillas aleatorias y proporcionar un único punto de modificación para cualquier ajuste de configuración. La instancia global `cfg` de la clase `Config` actúa como objeto de configuración compartido que todos los módulos del proyecto importan y consultan.

El diseño emplea `dataclasses` de Python para beneficiarse del tipado estático, la generación automática de `__repr__` legible y la posibilidad de validación en `__post_init__`, siguiendo el principio de *Configuration as Code* documentado en la literatura de ingeniería de software reproducible [Sculley et al., 2015].

---

## Fundamento Teórico

### Reproducibilidad en experimentos de ML

La reproducibilidad en sistemas de aprendizaje automático exige el control de toda fuente de aleatoriedad. El parámetro `RANDOM_SEED` se propaga a:

- `numpy.random.seed(seed)`
- `random.seed(seed)`
- `sklearn` (vía `random_state=seed` en todos los estimadores)
- `xgboost` (vía `seed=seed`)

Bajo estas condiciones, dos ejecuciones con idénticos datos e hiperparámetros producen resultados bit-a-bit equivalentes.

### Partición estratificada train / val / test

El dataset se divide según los ratios $r_\text{train}$, $r_\text{val}$, $r_\text{test}$ tales que:

$$r_\text{train} + r_\text{val} + r_\text{test} = 1$$

La partición es *estratificada* respecto a la variable objetivo para preservar la distribución de clases en cada subconjunto, lo cual es crítico dado el severo desequilibrio de clases presente en CICIDS2017 (la clase BENIGN supera el 80 % de las muestras).

### Filtrado de clases minoritarias

Se eliminan las clases que no superan *simultáneamente* dos umbrales:

$$n_k < N_\text{min} \quad \lor \quad \frac{n_k}{N} < r_\text{min}$$

donde $n_k$ es la cardinalidad de la clase $k$ y $N$ el total de muestras. Este doble criterio evita que clases con muy pocas muestras distorsionen las métricas de evaluación y dificulten el entrenamiento de clasificadores.

---

## Constantes Globales

### `ATTACK_GROUPS: Dict[str, str]`

**Descripción:** Diccionario de mapeo que agrupa las 15 etiquetas originales del dataset CICIDS2017 en 7 clases canónicas. La agregación reduce la fragmentación de clases semánticamente equivalentes (p. ej. las cinco variantes de DoS) y mejora la estimación estadística en clases minoritarias.

**Clases canónicas resultantes:** `BENIGN`, `DoS`, `DDoS`, `Scan`, `BruteForce`, `WebAttack`, `Botnet`.

### `BENIGN_LABEL: str`

**Descripción:** Etiqueta canónica de la clase de tráfico legítimo. Utilizada por `IsolationForestModel` para filtrar el subconjunto de entrenamiento del Stage 1 (solo muestras benignas).

### `CLASS_ORDER: List[str]`

**Descripción:** Orden canónico de las 7 clases (BENIGN primero, ataques en orden alfabético). Garantiza consistencia en matrices de confusión, curvas ROC y otras representaciones gráficas.

---

## Clases

### `IsolationForestConfig`

**Descripción:** Hiperparámetros por defecto del Isolation Forest (Stage 1 del pipeline en cascada).

**Parámetros:**

| Atributo | Tipo | Valor por defecto | Descripción |
|---|---|---|---|
| `n_estimators` | `int` | `200` | Número de árboles de aislamiento en el ensemble |
| `max_samples` | `str` | `"auto"` | Tamaño del subconjunto de muestreo por árbol; `"auto"` selecciona min(256, n_samples) |
| `contamination` | `float` | `0.15` | Fracción esperada de anomalías; define el umbral inicial de decisión |
| `max_features` | `float` | `1.0` | Fracción de features usadas para construir cada árbol |
| `bootstrap` | `bool` | `False` | Si `True`, el muestreo es con reemplazo |
| `n_jobs` | `int` | `-1` | Paralelismo; `-1` usa todos los núcleos disponibles |

**Notas:** El umbral de decisión se reoptimiza en `models/isolation_forest.py` mediante maximización de F-beta sobre el conjunto de validación, por lo que `contamination` actúa solo como inicialización y puede diferir del umbral final empleado en inferencia.

---

### `XGBoostConfig`

**Descripción:** Hiperparámetros por defecto del clasificador XGBoost (Stage 2, Pipeline A) y espacio de búsqueda para `RandomizedSearchCV`.

**Parámetros (valores por defecto):**

| Atributo | Tipo | Valor | Descripción |
|---|---|---|---|
| `n_estimators` | `int` | `300` | Número de árboles de boosting |
| `max_depth` | `int` | `6` | Profundidad máxima de cada árbol |
| `learning_rate` | `float` | `0.1` | Tasa de aprendizaje (shrinkage) $\eta$ |
| `subsample` | `float` | `0.9` | Fracción de muestras usadas por árbol |
| `colsample_bytree` | `float` | `0.9` | Fracción de features usadas por árbol |
| `early_stopping_rounds` | `int` | `20` | Paciencia para early stopping sobre `eval_metric` |
| `tree_method` | `str` | `"hist"` | Algoritmo de construcción de árboles; `"hist"` es eficiente en CPU y compatible con GPU |

**Espacio de búsqueda (`search_space`):**

```python
{
    "n_estimators":      [200, 300, 500],
    "max_depth":         [4, 6, 8],
    "learning_rate":     [0.05, 0.1, 0.2],
    "subsample":         [0.8, 1.0],
    "colsample_bytree":  [0.8, 1.0],
}
```

Se muestrean `n_iter_search = 20` combinaciones aleatorias con validación cruzada estratificada de `cv_folds = 3` particiones.

---

### `NeuralNetworkConfig`

**Descripción:** Hiperparámetros por defecto de la red neuronal NumPy (Stage 2, Pipeline B).

**Arquitectura:**

```
Input → StandardScaler
      → Dense(256) → BatchNorm → ReLU → Dropout(0.15)
      → Dense(128) → BatchNorm → ReLU → Dropout(0.15)
      → Dense(64)  → BatchNorm → ReLU → Dropout(0.10)
      → Dense(n_classes) → Softmax
```

**Parámetros:**

| Atributo | Tipo | Valor | Descripción |
|---|---|---|---|
| `hidden_layers` | `List[int]` | `[256, 128, 64]` | Número de neuronas por capa oculta |
| `dropout_rates` | `List[float]` | `[0.15, 0.15, 0.10]` | Tasa de Dropout por capa oculta |
| `use_batch_norm` | `bool` | `True` | Activa Batch Normalization en todas las capas ocultas |
| `learning_rate` | `float` | `1e-3` | Tasa de aprendizaje inicial de Adam |
| `beta1` | `float` | `0.9` | Decaimiento del momento de primer orden en Adam |
| `beta2` | `float` | `0.999` | Decaimiento del momento de segundo orden en Adam |
| `epsilon` | `float` | `1e-8` | Término de estabilidad numérica en Adam |
| `weight_decay` | `float` | `1e-4` | Coeficiente de regularización L2 |
| `epochs` | `int` | `100` | Número máximo de épocas de entrenamiento |
| `batch_size` | `int` | `512` | Tamaño del mini-batch |
| `label_smoothing` | `float` | `0.05` | Suavizado de etiquetas; reducido de 0.1 para no diluir la señal en clases con pocas muestras (p. ej. Botnet) |
| `early_stopping_patience` | `int` | `20` | Épocas sin mejora antes de detener el entrenamiento; aumentado de 10 para dar margen al LR scheduler |
| `early_stopping_min_delta` | `float` | `1e-4` | Mejora mínima considerada significativa |
| `use_class_weights` | `bool` | `True` | Pondera la pérdida de cada muestra por el inverso de la frecuencia de su clase, compensando el desequilibrio extremo (Botnet ≈ 0.1 %, BENIGN ≈ 83 %) |
| `lr_scheduler_patience` | `int` | `5` | Épocas sin mejora antes de reducir la LR (ReduceLROnPlateau) |
| `lr_scheduler_factor` | `float` | `0.5` | Factor multiplicativo al reducir la LR (`new_lr = lr × factor`) |
| `lr_min` | `float` | `1e-5` | Valor mínimo al que puede descender la LR |

**Mejoras v2:**

| Atributo | Tipo | Valor | Descripción |
|---|---|---|---|
| `use_balanced_batching` | `bool` | `True` | Si `True`, los índices de cada epoch se muestrean estratificadamente por clase (proporciones iguales) en lugar de permutarse aleatoriamente; evita batches homogéneos que anulan la señal de clases minoritarias |
| `focal_gamma` | `float` | `2.0` | Exponente γ de la Focal Loss: `focal_w = (1 − p_t)^γ`; penaliza más las muestras clasificadas con alta confianza errónea; se combina multiplicativamente con los pesos de clase |
| `oversample_min_ratio` | `float` | `0.0` | Fracción mínima respecto a la clase mayoritaria para clases minoritarias; si > 0, las clases por debajo del umbral se sobremuestrean con ruido gaussiano σ = 0.01 (en espacio escalado); `0.0` desactiva el oversampling |
| `lr_warmup_epochs` | `int` | `5` | Número de épocas de calentamiento lineal de la LR: desde `lr_min` hasta `learning_rate`; `lr_epoch = lr_min + (lr_init − lr_min) × epoch / lr_warmup_epochs` |
| `grad_clip_norm` | `float` | `1.0` | Umbral de norma L2 global para gradient clipping; si la norma de todos los gradientes supera este valor, se reescalan uniformemente antes de la actualización Adam; `0.0` desactiva el clipping |
| `confidence_threshold` | `float` | `0.0` | Umbral de confianza mínima en inferencia: si `max_prob < threshold`, se descarta la clase con mayor probabilidad y se usa la segunda mejor; `0.0` desactiva el mecanismo |

---

### `Config`

**Descripción:** Clase de configuración global del proyecto. Agrega todas las sub-configuraciones de modelos y los parámetros de pipeline, datos y reproducibilidad. La instancia global `cfg` es importada por el resto de módulos.

**Parámetros:**

| Atributo | Tipo | Valor por defecto | Descripción |
|---|---|---|---|
| `DATA_PATH` | `Path` | `Path("data")` | Directorio con los CSVs de CICIDS2017 |
| `OUTPUT_PATH` | `Path` | `Path("output")` | Directorio raíz de salida del experimento |
| `RANDOM_SEED` | `int` | `42` | Semilla global de aleatoriedad |
| `SAMPLE_SIZE` | `int` | `200_000` | Muestras estratificadas; `0` = dataset completo |
| `TRAIN_RATIO` | `float` | `0.70` | Fracción del dataset para entrenamiento |
| `VAL_RATIO` | `float` | `0.15` | Fracción del dataset para validación |
| `TEST_RATIO` | `float` | `0.15` | Fracción del dataset para test |
| `MIN_CLASS_SAMPLES` | `int` | `100` | Umbral absoluto mínimo de muestras por clase |
| `MIN_CLASS_RATIO` | `float` | `0.0005` | Umbral relativo mínimo (fracción del total) |
| `BETA` | `float` | `2.0` | Parámetro F-beta para optimización del umbral del IF; el valor `2.0` pondera más el recall, favoreciendo la detección de ataques sobre la reducción de falsas alarmas |
| `isolation_forest` | `IsolationForestConfig` | — | Sub-configuración del Isolation Forest |
| `xgboost` | `XGBoostConfig` | — | Sub-configuración del clasificador XGBoost |
| `neural_network` | `NeuralNetworkConfig` | — | Sub-configuración de la red neuronal NumPy |

**Propiedades (paths derivados):**

| Propiedad | Retorna |
|---|---|
| `eda_path` | `OUTPUT_PATH / "eda"` |
| `pipeline_a_path` | `OUTPUT_PATH / "pipeline_a"` |
| `pipeline_b_path` | `OUTPUT_PATH / "pipeline_b"` |
| `comparison_path` | `OUTPUT_PATH / "comparison"` |
| `results_file` | `OUTPUT_PATH / "results.txt"` |

**Métodos:**

#### `create_output_dirs() -> None`

**Descripción:** Crea la estructura completa de directorios de salida (`eda/`, `pipeline_a/`, `pipeline_b/`, `comparison/`) de forma idempotente.  
**Notas:** Debe invocarse al inicio de `main.py`, antes de cualquier escritura en disco.

**Retorna:** `None`

#### `__post_init__() -> None`

**Descripción:** Ejecutado automáticamente por el mecanismo de `dataclass`. Normaliza `DATA_PATH` y `OUTPUT_PATH` a objetos `Path` e invoca `_validate_ratios` para verificar que los ratios de partición sumen exactamente 1.0.

---

### `_validate_ratios(train, val, test) -> None`

**Descripción:** Función de validación interna que verifica la consistencia de la partición del dataset.

**Parámetros:**
- `train` — fracción de entrenamiento ∈ (0, 1)
- `val` — fracción de validación ∈ (0, 1)
- `test` — fracción de test ∈ (0, 1)

**Raises:** `ValueError` si |train + val + test − 1| > 1×10⁻⁹

---

## Uso Típico

```python
from config import cfg, ATTACK_GROUPS, BENIGN_LABEL

# Acceso a rutas
data_dir = cfg.DATA_PATH          # Path("data")
output_dir = cfg.OUTPUT_PATH      # Path("output")

# Crear directorios de salida
cfg.create_output_dirs()

# Modificar desde main.py (p. ej. vía argparse)
cfg.DATA_PATH = Path(args.data_path)
cfg.RANDOM_SEED = args.seed
cfg.BETA = args.beta

# Acceso a hiperparámetros de modelos
n_trees = cfg.isolation_forest.n_estimators     # 200
lr = cfg.neural_network.learning_rate           # 0.001
```

---

## Dependencias Internas

Este módulo no depende de ningún otro módulo del proyecto. Es la hoja raíz del grafo de dependencias y debe importarse antes que cualquier otro componente.

---

## Referencias

- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. Advances in Neural Information Processing Systems, 28.
- Python Software Foundation. (2024). *dataclasses — Data Classes*. Python 3.10 Documentation.
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward generating a new intrusion detection dataset and intrusion traffic characterization*. ICISSP, 108–116.
