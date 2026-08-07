# Módulo `pipelines.pipeline_b`

## Descripción General

El módulo `pipeline_b` implementa la segunda de las dos arquitecturas de detección comparadas en el experimento: un pipeline en cascada que combina el mismo detector de anomalías no supervisado (Isolation Forest) del Pipeline A en el Estadio 1 con un perceptrón multicapa implementado íntegramente en NumPy (Stage 2). Su clase principal, `PipelineB`, extiende la interfaz abstracta `BasePipeline` y proporciona una implementación completa de los métodos `train`, `predict`, `evaluate` y `save`.

El Pipeline B constituye el componente requerido del proyecto: su red neuronal implementada sin frameworks de *deep learning* permite una comparación directa con el clasificador XGBoost del Pipeline A, aislando el efecto del clasificador de Estadio 2 sobre las métricas globales del sistema.

## Fundamento Teórico

### Arquitectura en cascada

La arquitectura en cascada es idéntica a la del Pipeline A en su Estadio 1:

$$
\hat{y} = \begin{cases}
\text{BENIGN}            & \text{si } s(x) \geq \tau^* \\
f_{\text{MLP}}(x)        & \text{si } s(x) < \tau^*
\end{cases}
$$

donde $s(x)$ es la *decision function* del Isolation Forest y $\tau^*$ es el umbral óptimo determinado mediante maximización del $F_\beta$-score sobre el conjunto de validación (véase `models.isolation_forest`).

### Stage 2 — Clasificación multiclase con MLP NumPy

El clasificador de Estadio 2 es un perceptrón multicapa entrenado exclusivamente sobre las muestras marcadas como anómalas por el IF. La arquitectura fija es:

$$
\text{Input} \xrightarrow{W_1, b_1} \text{Dense}(128) \xrightarrow{\text{BN}} \text{ReLU} \xrightarrow{\text{Drop}(0.3)} \xrightarrow{W_2, b_2} \text{Dense}(64) \xrightarrow{\text{ReLU}} \xrightarrow{\text{Drop}(0.2)} \xrightarrow{W_3, b_3} \text{Dense}(K) \xrightarrow{\text{Softmax}} \hat{p}
$$

donde $K$ es el número de clases de ataque del espacio canónico definido en `CLASS_ORDER` (excluyendo `BENIGN`), que permanece constante independientemente de qué clases aparezcan en las muestras anómalas enrutadas por el Stage 1. Esta decisión garantiza que la dimensión de salida del softmax y el mapeo de clases del `LabelEncoder` sean siempre coherentes con el espacio de predicción completo del sistema.

La función de pérdida es la entropía cruzada con *label smoothing* $\varepsilon = 0.1$:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} \tilde{y}_{ik} \log \hat{p}_{ik}, \quad \tilde{y}_{ik} = (1 - \varepsilon)\,\delta_{k,c_i} + \frac{\varepsilon}{K}
$$

La optimización se realiza con Adam (Kingma & Ba, 2015) con corrección de sesgo y decaimiento de pesos $L_2$:

$$
\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

### Instanciación diferida de la red

A diferencia del Pipeline A —donde `XGBoostClassifier` puede construirse sin conocer las dimensiones de los datos—, `NeuralNetwork` requiere `n_features` y `n_classes` en su constructor. Estos valores sólo son conocidos tras el enrutamiento de Estadio 1, por lo que la red se instancia de forma **diferida dentro de `train()`** una vez determinado el subconjunto de muestras anómalas.

## Funciones / Clases

### `PipelineB(config, feature_names)`

**Descripción:** Clase orquestadora del Pipeline B. Instancia `IsolationForestModel` en `__init__` y crea `NeuralNetwork` de forma diferida en `train()` una vez conocidas las dimensiones del subconjunto de ataque.

**Parámetros:**
- `config` — instancia de `Config`; si es `None` se usa la instancia global `cfg`.
- `feature_names` — lista de nombres de columnas del dataset; reservado para uso futuro (el MLP no produce importancia de features).

---

### `train(X_train, y_train, X_val, y_val, *, beta, skip_tuning) -> PipelineB`

**Descripción:** Entrena los dos estadios del pipeline de forma secuencial. El Estadio 1 (IF) se entrena sobre todas las muestras BENIGN y optimiza el umbral sobre el conjunto de validación. Las muestras anómalas del conjunto de entrenamiento se enrutan al Estadio 2, donde se instancia y entrena la red neuronal NumPy.

**Parámetros:**
- `X_train` — matriz de features de entrenamiento, shape `(n_train, n_features)`.
- `y_train` — etiquetas de clase canónicas del conjunto de entrenamiento, shape `(n_train,)`.
- `X_val` — matriz de features de validación, shape `(n_val, n_features)`.
- `y_val` — etiquetas de clase del conjunto de validación, shape `(n_val,)`.
- `beta` — peso F-beta para la optimización del umbral del IF.
- `skip_tuning` — aceptado por compatibilidad con `BasePipeline`; sin efecto sobre la red neuronal (no se define espacio de búsqueda de hiperparámetros).

**Retorna:** `self`.

**Notas:** Si el Estadio 1 no enruta ninguna muestra anómala, se lanza `RuntimeError`. Si el conjunto de validación no contiene muestras anómalas, se usa el 10 % inicial del conjunto de entrenamiento anómalo como pseudo-validación para satisfacer el requisito de *early stopping* de la red. El número de clases del MLP se fija al espacio canónico de ataque (`CLASS_ORDER` menos `BENIGN_LABEL`), y el `LabelEncoder` interno de la red se inicializa sobre ese mismo espacio antes del entrenamiento, garantizando consistencia incluso cuando alguna clase de ataque no aparece en las muestras anómalas del conjunto de entrenamiento.

---

### `predict(X) -> np.ndarray`

**Descripción:** Ejecuta la inferencia completa en cascada. El IF detecta anomalías; las muestras anómalas se pasan al MLP para clasificar el tipo de ataque; las muestras no anómalas reciben la etiqueta `'BENIGN'` directamente.

**Parámetros:**
- `X` — matriz de features, shape `(n_samples, n_features)`.

**Retorna:** `np.ndarray` de cadenas, shape `(n_samples,)`.

**Complejidad:** $O(n \cdot d)$ para el IF más $O(m \cdot L \cdot H)$ para el MLP, donde $m \leq n$ es el número de muestras anómalas, $L$ el número de capas y $H$ la anchura máxima.

---

### `evaluate(X_test, y_test) -> Dict[str, float]`

**Descripción:** Calcula las métricas de clasificación *end-to-end* sobre el conjunto de test. Invoca `predict()` internamente.

**Parámetros:**
- `X_test` — matriz de features del conjunto de test, shape `(n_test, n_features)`.
- `y_test` — etiquetas de referencia, shape `(n_test,)`.

**Retorna:** Diccionario con las siguientes claves:

| Clave           | Descripción                                       |
|-----------------|---------------------------------------------------|
| `accuracy`      | Fracción de predicciones correctas                |
| `f1_macro`      | F1 promediado macro (peso igual por clase)        |
| `f1_weighted`   | F1 ponderado por soporte de clase                 |
| `if_precision`  | Precisión del IF en la detección de anomalías     |
| `if_recall`     | Recall del IF (tasa de detección de ataques)      |
| `if_fpr`        | Tasa de falsos positivos del IF                   |
| `if_fnr`        | Tasa de falsos negativos del IF                   |

---

### `save(directory) -> None`

**Descripción:** Persiste los dos componentes del pipeline en subdirectorios separados de `directory`:
- `<directory>/isolation_forest/` — modelo IF serializado con joblib y array NumPy del umbral.
- `<directory>/neural_network/` — pesos de la red en `nn_weights.npz`, estadísticas BatchNorm en `nn_bn_running.npz`, y `LabelEncoder` en `nn_label_encoder.joblib`.

**Parámetros:**
- `directory` — ruta destino; se crea recursivamente si no existe.

---

### Propiedad `if_model -> IsolationForestModel`

**Descripción:** Retorna el modelo de Estadio 1. Permite al código externo acceder al IF sin romper el encapsulamiento mediante atributos privados.

---

### Propiedad `stage2_classifier -> NeuralNetwork`

**Descripción:** Retorna el clasificador de Estadio 2. Lanza `RuntimeError` si el pipeline no ha sido entrenado aún. Permite al código externo acceder al MLP (p. ej., para obtener `predict_proba` y `classes_`) sin acceder directamente a `_nn`.

---

### `get_training_history() -> Dict[str, List[float]]`

**Descripción:** Retorna el historial de entrenamiento por epoch de la red neuronal, con claves `'train_loss'`, `'train_acc'`, `'val_loss'`, `'val_acc'`. Usado por el módulo de visualización para generar `training_curves.png`.

---

### `get_if_operating_point() -> dict`

**Descripción:** Retorna el diccionario de diagnóstico del Estadio 1: umbral activo, $\beta$, $F_\beta$-score, precisión, recall, FPR, FNR y conteos TP/FP/TN/FN.

## Diferencias con Pipeline A

| Aspecto                   | Pipeline A                            | Pipeline B                                    |
|---------------------------|---------------------------------------|-----------------------------------------------|
| Clasificador Stage 2      | `XGBoostClassifier`                   | `NeuralNetwork` (NumPy puro)                  |
| Instanciación Stage 2     | En `__init__`                         | Diferida en `train()` (requiere dimensiones)  |
| Ajuste de hiperparámetros | `RandomizedSearchCV` (omitible)       | No aplica; `skip_tuning` sin efecto           |
| Persistencia Stage 2      | joblib + `LabelEncoder`               | `numpy.savez` + joblib `LabelEncoder`         |
| Accessors adicionales     | `get_feature_importance()`            | `get_training_history()`                      |
| Propiedades públicas      | `if_model`, `stage2_classifier`       | `if_model`, `stage2_classifier`               |
| Curvas de entrenamiento   | No disponibles                        | `history` → `training_curves.png`             |

## Dependencias Internas

- [`config`](../../config.py) — `Config`, `cfg`, `BENIGN_LABEL`, `CLASS_ORDER`.
- [`models.isolation_forest`](../../models/isolation_forest.py) — `IsolationForestModel`.
- [`models.neural_network`](../../models/neural_network.py) — `NeuralNetwork`.
- [`pipelines.base`](../../pipelines/base.py) — `BasePipeline`.

## Referencias

- Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization*. ICLR 2015.
- Ioffe, S., & Szegedy, C. (2015). *Batch normalization: Accelerating deep network training*. ICML 2015.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation forest*. IEEE ICDM 2008, 413–422.
- Müller, R., Kornblith, S., & Hinton, G. (2019). *When does label smoothing help?* NeurIPS 2019.
