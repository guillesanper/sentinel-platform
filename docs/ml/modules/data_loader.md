# Módulo `data.loader`

## Descripción General

El módulo `data.loader` constituye la capa de ingesta y preprocesamiento del sistema. Implementa el ciclo completo de carga del dataset CICIDS2017: lectura y concatenación de múltiples ficheros CSV, normalización de artefactos numéricos, detección y eliminación automática de clases hiper-minoritarias, muestreo estratificado, análisis exploratorio de datos (EDA) y partición reproducible en subconjuntos de entrenamiento, validación y test. Todas las operaciones se registran a través del sistema de logging definido en `utils.logger`, garantizando la trazabilidad íntegra del experimento.

---

## Fundamento Teórico

### Mapeo de etiquetas

El dataset CICIDS2017 contiene 15 etiquetas de ataque con distribución altamente desequilibrada. El módulo aplica la función de agrupación $g: \mathcal{L}_{15} \rightarrow \mathcal{C}_7$ definida en `config.ATTACK_GROUPS`, colapsando las 15 clases originales en 7 categorías canónicas: `BENIGN`, `DoS`, `DDoS`, `Scan`, `BruteForce`, `WebAttack` y `Botnet`.

### Filtrado de clases minoritarias

Sea $n_k$ el número de muestras de la clase $k$ y $N$ el total del dataset. La clase $k$ se elimina si:

$$n_k < n_{\min} \quad \text{OR} \quad \frac{n_k}{N} < r_{\min}$$

donde $n_{\min}$ = `MIN_CLASS_SAMPLES` y $r_{\min}$ = `MIN_CLASS_RATIO`. El filtrado precede al muestreo estratificado para impedir que clases residuales generen splits inválidos.

### Muestreo estratificado

Dado un tamaño objetivo $S$ y una proporción $f = S/N$, el muestreo estratificado preserva la distribución de clases original extrayendo $\lfloor f \cdot n_k \rfloor$ muestras de cada clase $k$. Un ajuste posterior corrige el déficit de redondeo mediante submuestreo aleatorio del conjunto no seleccionado.

### Selección de features discriminativas para EDA

Para las curvas de distribución por clase (`feature_distributions.png`), la selección de las top-`_DIST_N_FEATURES` features se realiza mediante el estadístico F de ANOVA univariante (`sklearn.feature_selection.f_classif`). El estadístico F mide la varianza entre grupos (clases) normalizada por la varianza intra-grupo:

$$F_j = \frac{\text{SS}_{\text{between}} / (K-1)}{\text{SS}_{\text{within}} / (N-K)}$$

donde $K$ es el número de clases canónicas y $N$ el número de muestras. Las features con mayor $F_j$ son las que mejor separan las distribuciones entre clases, lo que las convierte en las más informativas para el análisis exploratorio de un sistema de detección de intrusiones. Este criterio es más relevante que la varianza pura, que no captura la separabilidad inter-clase.

---

### Partición train/val/test

La partición se realiza en dos etapas de *stratified splitting* (scikit-learn `train_test_split`):

1. $(D_{\text{train+val}}, D_{\text{test}})$: fracción de test $= r_{\text{test}}$.
2. $(D_{\text{train}}, D_{\text{val}})$: fracción de validación renormalizada $= r_{\text{val}} / (r_{\text{train}} + r_{\text{val}})$.

La estratificación en ambos pasos garantiza que la distribución de clases en cada subset es proporcional a la del dataset completo.

---

## Constantes del Módulo

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `LABEL_COL` | `"Label"` | Nombre de la columna de etiquetas tras normalización |
| `_CORR_N_FEATURES` | `20` | Número de features (por varianza) usadas en el heatmap de correlación |
| `_DIST_N_FEATURES` | `6` | Número de features (por ANOVA F-statistic) usadas en las curvas de distribución |

---

## Funciones Públicas

### `load_cicids2017(data_path: Path, config: Config = cfg) -> pd.DataFrame`

**Descripción:** Lee todos los ficheros `.csv` presentes en `data_path`, normaliza los nombres de columna (elimina espacios iniciales/finales), concatena los DataFrames parciales y aplica la función de mapeo de etiquetas.

**Parámetros:**
- `data_path` — ruta al directorio que contiene los ficheros CSV de CICIDS2017
- `config` — instancia de configuración global (por defecto `cfg`)

**Retorna:** `pd.DataFrame` concatenado con etiquetas canónicas en la columna `Label`.

**Notas:** Antes de ejecutar el glob, valida que `data_path` exista como directorio: lanza `FileNotFoundError` si la ruta no existe y `NotADirectoryError` si apunta a un fichero en lugar de un directorio. Lanza `FileNotFoundError` si no se encuentra ningún fichero `.csv`. Los ficheros se cargan en orden lexicográfico con barra de progreso `tqdm`.

---

### `clean_dataset(df: pd.DataFrame) -> pd.DataFrame`

**Descripción:** Elimina artefactos numéricos del dataset bruto en tres pasos secuenciales:
1. Sustitución de valores `±Inf` por `NaN`.
2. Eliminación de filas con valores `NaN` en columnas de features.
3. Eliminación de filas duplicadas exactas.
4. Conversión de todas las features a `float32`.

**Parámetros:**
- `df` — DataFrame bruto con columna `Label`

**Retorna:** DataFrame limpio con tipos de datos homogéneos e índice reseteado.

**Notas:** Tras la eliminación de NaN/Inf, si el porcentaje de filas eliminadas supera el 20% del total original se emite un `logger.warning`, dado que en el dataset CICIDS2017 se espera que esta tasa sea inferior al 5% con la codificación CSV estándar.

---

### `filter_minority_classes(df: pd.DataFrame, config: Config = cfg) -> Tuple[pd.DataFrame, List[str]]`

**Descripción:** Identifica y elimina las clases que no superan los umbrales mínimos de representatividad definidos en la configuración.

**Parámetros:**
- `df` — DataFrame limpio con columna `Label`
- `config` — instancia de configuración con `MIN_CLASS_SAMPLES` y `MIN_CLASS_RATIO`

**Retorna:** Tupla `(df_filtrado, clases_eliminadas)`. Emite `logger.warning` por cada clase eliminada.

**Notas:** Debe ejecutarse antes de `stratified_sample` para evitar que clases residuales hereden cuotas de muestreo.

---

### `stratified_sample(df: pd.DataFrame, config: Config = cfg) -> pd.DataFrame`

**Descripción:** Extrae una muestra estratificada de tamaño `config.SAMPLE_SIZE`. Si el dataset es igual o más pequeño que `SAMPLE_SIZE`, o si `SAMPLE_SIZE == 0`, devuelve el DataFrame sin modificar.

**Parámetros:**
- `df` — DataFrame filtrado con columna `Label`
- `config` — instancia de configuración con `SAMPLE_SIZE` y `RANDOM_SEED`

**Retorna:** DataFrame muestreado (o el original), mezclado y con índice reseteado.

---

### `run_eda(df: pd.DataFrame, config: Config = cfg) -> None`

**Descripción:** Genera y guarda las tres figuras de análisis exploratorio en `config.eda_path`:

| Fichero | Contenido |
|---------|-----------|
| `class_distribution.png` | Gráfico de barras + diagrama de sectores por clase |
| `feature_correlation.png` | Mapa de calor de correlaciones de Pearson para el top-`_CORR_N_FEATURES` features por varianza |
| `feature_distributions.png` | Curvas KDE por clase para el top-`_DIST_N_FEATURES` features seleccionadas por ANOVA F-statistic (mayor poder discriminativo entre las 7 clases canónicas) |

**Parámetros:**
- `df` — DataFrame muestreado con columna `Label`
- `config` — instancia de configuración con `eda_path`

**Retorna:** `None`. Efectos secundarios: escritura de tres ficheros PNG en `output/eda/`.

---

### `split_dataset(df: pd.DataFrame, config: Config = cfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

**Descripción:** Particiona el dataset en tres subsets mediante doble *stratified split* reproducible.

**Parámetros:**
- `df` — DataFrame final (limpio, filtrado, muestreado) con columna `Label`
- `config` — instancia de configuración con `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` y `RANDOM_SEED`

**Retorna:** Tupla `(train_df, val_df, test_df)` con índices reseteados.

**Complejidad:** O(N) — dos pasadas lineales sobre el dataset.

**Notas:** Registra en `DEBUG` la distribución de clases por split para verificación de reproducibilidad.

---

## Dependencias Internas

| Módulo | Uso |
|--------|-----|
| `config` | `ATTACK_GROUPS`, `BENIGN_LABEL`, `Config`, `cfg` |
| `utils.logger` | `get_logger` — trazabilidad de todas las operaciones |

---

## Referencias

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP*, 108–116.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51–56.
