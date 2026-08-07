# Módulo `models/isolation_forest`

## Descripción General

Este módulo implementa el primer estadio (*Stage 1*) del pipeline de detección de intrusiones en cascada. La clase `IsolationForestModel` encapsula un detector de anomalías basado en el algoritmo Isolation Forest, entrenado exclusivamente sobre tráfico de red benigno (etiqueta `BENIGN`) del conjunto CICIDS2017. Su función es filtrar el tráfico entrante en dos categorías: *normal* —clasificado directamente como BENIGN— y *anómalo* —remitido al clasificador de Stage 2 para su tipificación específica de ataque—. El módulo expone además un mecanismo de optimización del umbral de decisión basado en la métrica F-beta, lo que permite ajustar el compromiso entre tasa de falsos positivos y tasa de falsos negativos según los requisitos operativos del sistema.

---

## Fundamento Teórico

### Isolation Forest

El algoritmo Isolation Forest (Liu et al., 2008) fundamenta la detección de anomalías en el principio de *aislabilidad*: las muestras atípicas son estadísticamente más fáciles de separar del resto de la distribución mediante particiones aleatorias del espacio de características. Formalmente, dado un conjunto de entrenamiento $\mathcal{X} \subset \mathbb{R}^d$, el algoritmo construye un ensamble de $T$ árboles de aislamiento (*iTrees*). Cada árbol $h_t$ particiona recursivamente el espacio seleccionando en cada nodo un atributo $q \in \{1,\dots,d\}$ y un valor de corte $p$ uniformemente distribuido en $[\min(q), \max(q)]$ sobre las muestras presentes.

La profundidad de aislamiento $h_t(\mathbf{x})$ de una muestra $\mathbf{x}$ en el árbol $t$ corresponde al número de aristas recorridas desde la raíz hasta el nodo hoja donde $\mathbf{x}$ queda aislada. La puntuación de anomalía (*anomaly score*) se define como:

$$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}[h(\mathbf{x})]}{c(n)}}$$

donde $\mathbb{E}[h(\mathbf{x})]$ es la longitud de camino media sobre el ensamble y $c(n)$ es el factor de normalización correspondiente a la longitud media de camino en un árbol de búsqueda binaria exitosa sobre $n$ muestras:

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

siendo $H(k) = \ln(k) + \gamma_{\text{Euler}}$ la función armónica. Los valores de $s$ próximos a 1 indican anomalías; los próximos a 0.5 indican normalidad.

En la implementación de scikit-learn, `decision_function` devuelve la diferencia entre la longitud de camino media normalizada y el valor de referencia $c(n)$, con el signo convencional: **puntuaciones positivas corresponden a inliers (BENIGN), puntuaciones negativas a outliers (ataque)**.

### Optimización del Umbral F-beta

El umbral por defecto $\tau = 0$ de scikit-learn maximiza la separación inlier/outlier en el conjunto de entrenamiento; sin embargo, puede ser subóptimo para el conjunto de validación o para un compromiso FP/FN específico. El módulo implementa una búsqueda exhaustiva sobre una malla de 300 candidatos $\{\tau_i\}_{i=1}^{300}$ uniformemente espaciados en el rango de puntuaciones del conjunto de validación:

$$\hat{\tau} = \arg\max_{\tau} F_\beta(\tau), \quad F_\beta = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

La regla de decisión binaria es:

$$\hat{y}(\mathbf{x}) = \begin{cases} 1 \text{ (anomalía)} & \text{si } s(\mathbf{x}) < \hat{\tau} \\ 0 \text{ (BENIGN)} & \text{si } s(\mathbf{x}) \geq \hat{\tau} \end{cases}$$

Con $\beta > 1$ la métrica pondera más el recall (se penalizan los falsos negativos, es decir, los ataques no detectados). Con $\beta < 1$ se pondera más la precisión (se penalizan los falsos positivos, reduciendo falsas alarmas).

---

## Clases y Métodos

### `IsolationForestModel`

Clase principal que encapsula el ciclo de vida completo del detector de anomalías.

**Constructor: `__init__(config: IsolationForestConfig, random_seed: int = 42) -> None`**

**Parámetros:**
- `config` — instancia de `IsolationForestConfig` con hiperparámetros del modelo (ver `config.py`)
- `random_seed` — semilla de reproducibilidad global

---

### `fit(X_train, y_train) -> IsolationForestModel`

**Descripción:** Filtra `X_train` para retener únicamente las filas con etiqueta `BENIGN` y ajusta el Isolation Forest sobre dicho subconjunto. El modelo nunca observa muestras de ataque durante el entrenamiento, lo que implementa el paradigma de detección de novedad (*novelty detection*).

**Parámetros:**
- `X_train : np.ndarray` — matriz de características de entrenamiento, shape `(n_samples, n_features)`
- `y_train : np.ndarray` — etiquetas de clase (strings del esquema canónico de 7 clases), shape `(n_samples,)`

**Retorna:** `self` (permite encadenamiento de métodos)

**Notas:** El umbral inicial tras el ajuste es $\tau = 0.0$ (valor por defecto de scikit-learn). Se recomienda llamar a `optimize_threshold` a continuación.

---

### `optimize_threshold(X_val, y_val, beta) -> float`

**Descripción:** Ejecuta la búsqueda del umbral óptimo F-beta sobre el conjunto de validación y actualiza el estado interno del modelo. Registra el punto de operación completo (precision, recall, FPR, FNR) en `_threshold_info`.

**Parámetros:**
- `X_val : np.ndarray` — características del conjunto de validación
- `y_val : np.ndarray` — etiquetas de clase del conjunto de validación
- `beta : float` — parámetro de peso del recall en F-beta (default `2.0`)

**Retorna:** `float` — valor del umbral óptimo $\hat{\tau}$

**Complejidad:** $O(300 \cdot n_{\text{val}})$ para la evaluación de candidatos

---

### `predict_binary(X) -> np.ndarray`

**Descripción:** Aplica el umbral activo $\hat{\tau}$ a las puntuaciones `decision_function` para producir predicciones binarias.

**Parámetros:**
- `X : np.ndarray` — matriz de características, shape `(n_samples, n_features)`

**Retorna:** `np.ndarray` de `int`, shape `(n_samples,)` — `1` = anomalía (enrutada a Stage 2), `0` = BENIGN

---

### `get_operating_point_info() -> dict`

**Descripción:** Devuelve un diccionario con las métricas de rendimiento calculadas en el umbral activo durante la última llamada a `optimize_threshold`.

**Retorna:** `dict` con las claves:

| Clave | Tipo | Descripción |
|-------|------|-------------|
| `threshold` | `float` | Valor de $\hat{\tau}$ activo |
| `beta` | `float` | Valor de beta usado en la optimización |
| `f{beta}_score` | `float` | F-beta en el umbral óptimo |
| `precision` | `float` | $\text{TP} / (\text{TP} + \text{FP})$ |
| `recall` | `float` | $\text{TP} / (\text{TP} + \text{FN})$ |
| `fpr` | `float` | $\text{FP} / (\text{FP} + \text{TN})$ — tasa de falsa alarma |
| `fnr` | `float` | $\text{FN} / (\text{FN} + \text{TP})$ — tasa de omisión |
| `tp`, `fp`, `tn`, `fn` | `int` | Componentes de la matriz de confusión binaria |

---

### `save(directory: Path) -> None`

**Descripción:** Serializa el objeto `IsolationForest` con `joblib` y el umbral como array NumPy. Crea el directorio si no existe.

**Archivos generados:**
- `isolation_forest.joblib` — modelo sklearn serializado
- `if_threshold.npy` — umbral $\hat{\tau}$ como array de un elemento

---

### `load(directory: Path) -> IsolationForestModel`

**Descripción:** Restaura el modelo y el umbral desde los archivos generados por `save`. Retorna `self` para permitir encadenamiento.

---

## Dependencias Internas

- [`config`](../../config.py) — `IsolationForestConfig`, `BENIGN_LABEL`
- [`utils/logger`](../utils/logger.md) — configuración del sistema de logging

## Referencias

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. En *Proceedings of the 8th IEEE International Conference on Data Mining* (ICDM), pp. 413–422. IEEE. https://doi.org/10.1109/ICDM.2008.17
- van Stein, B., & Bäck, T. (2020). *FAST-AD: Fast Anomaly Detection via the Isolation Forest algorithm*. arXiv:2002.11538.
- Scikit-learn developers (2024). `sklearn.ensemble.IsolationForest`. Scikit-learn 1.4 documentation.
