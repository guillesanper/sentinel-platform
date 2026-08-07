# Módulo `pipelines.pipeline_a`

## Descripción General

El módulo `pipeline_a` implementa la primera de las dos arquitecturas de detección comparadas en el experimento: un pipeline en cascada que combina un detector de anomalías no supervisado (Isolation Forest) en el Estadio 1 con un clasificador supervisado multiclase (XGBoost) en el Estadio 2. Su clase principal, `PipelineA`, extiende la interfaz abstracta `BasePipeline` y proporciona una implementación completa de los métodos `train`, `predict`, `evaluate` y `save`.

El pipeline constituye el modelo de referencia del proyecto (Pipeline A): su rendimiento en las métricas F1-macro, accuracy y AUC sirve como línea base para valorar la contribución de la red neuronal NumPy implementada en Pipeline B.

## Fundamento Teórico

### Arquitectura en cascada

La arquitectura en cascada divide la clasificación en dos etapas complementarias con objetivos distintos:

$$
\hat{y} = \begin{cases}
\text{BENIGN}          & \text{si } s(x) \geq \tau^* \\
f_{\text{XGB}}(x)      & \text{si } s(x) < \tau^*
\end{cases}
$$

donde $s(x)$ es la *decision function* del Isolation Forest y $\tau^*$ es el umbral óptimo determinado mediante maximización del $F_\beta$-score sobre el conjunto de validación (véase `models.isolation_forest`).

### Stage 2 — Clasificación multiclase con XGBoost

XGBoost (`multi:softprob`) resuelve la clasificación de ataques mediante un ensamblado de árboles de decisión con descenso de gradiente. La función de pérdida optimizada es la entropía cruzada multiclase:

$$
\mathcal{L} = -\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik}
$$

El desequilibrio entre clases de ataque se compensa mediante pesos de muestra inversamente proporcionales a la frecuencia de clase (`_compute_sample_weights` en `XGBoostClassifier`). El ajuste de hiperparámetros se realiza mediante `RandomizedSearchCV` con validación cruzada estratificada de 3 pliegues sobre las muestras anómalas del conjunto de entrenamiento.

## Funciones / Clases

### `PipelineA(config, feature_names)`

**Descripción:** Clase orquestadora del Pipeline A. Instancia y coordina `IsolationForestModel` y `XGBoostClassifier` siguiendo la arquitectura en cascada.

**Parámetros:**
- `config` — instancia de `Config`; si es `None` se usa la instancia global `cfg`.
- `feature_names` — lista de nombres de columnas del dataset; usada para etiquetar la importancia de features en la visualización.

---

### `train(X_train, y_train, X_val, y_val, *, beta, skip_tuning) -> PipelineA`

**Descripción:** Entrena los dos estadios del pipeline de forma secuencial. Primero ajusta el Isolation Forest sobre las muestras BENIGN del conjunto de entrenamiento y optimiza el umbral de decisión sobre el conjunto de validación. A continuación, filtra las muestras anómalas y las pasa al clasificador XGBoost para el entrenamiento supervisado.

**Parámetros:**
- `X_train` — matriz de features de entrenamiento, shape `(n_train, n_features)`.
- `y_train` — etiquetas de clase canónicas del conjunto de entrenamiento, shape `(n_train,)`.
- `X_val` — matriz de features de validación, shape `(n_val, n_features)`.
- `y_val` — etiquetas de clase del conjunto de validación, shape `(n_val,)`.
- `beta` — peso F-beta para la optimización del umbral del IF; `beta > 1` penaliza falsos negativos, `beta < 1` penaliza falsos positivos.
- `skip_tuning` — si es `True`, omite la búsqueda aleatoria de hiperparámetros en XGBoost y entrena con los valores de `config.xgboost`.

**Retorna:** `self` (permite encadenamiento fluido).

**Notas:** Si el Estadio 1 no enruta ninguna muestra anómala al Estadio 2, se lanza `RuntimeError`. Si el conjunto de validación no contiene muestras anómalas, se utiliza el 10 % inicial del conjunto de entrenamiento anómalo como pseudo-validación para satisfacer el requisito de `eval_set` de XGBoost.

---

### `predict(X) -> np.ndarray`

**Descripción:** Ejecuta la inferencia completa en cascada sobre la matriz de features `X`. Aplica el Isolation Forest para identificar muestras anómalas y reenvía únicamente éstas al clasificador XGBoost; las muestras no anómalas reciben la etiqueta `'BENIGN'` directamente.

**Parámetros:**
- `X` — matriz de features, shape `(n_samples, n_features)`.

**Retorna:** `np.ndarray` de cadenas, shape `(n_samples,)`, con las etiquetas de clase predichas del esquema canónico de 7 clases.

**Complejidad:** $O(n \cdot d)$ para el IF más $O(m \cdot T \cdot d)$ para XGBoost, donde $m \leq n$ es el número de muestras anómalas y $T$ el número de árboles.

---

### `evaluate(X_test, y_test) -> Dict[str, float]`

**Descripción:** Calcula las métricas de clasificación *end-to-end* sobre el conjunto de test. Invoca `predict()` internamente y compara contra las etiquetas de referencia.

**Parámetros:**
- `X_test` — matriz de features del conjunto de test, shape `(n_test, n_features)`.
- `y_test` — etiquetas de referencia del conjunto de test, shape `(n_test,)`.

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
- `<directory>/xgboost/` — modelo XGBoost serializado con joblib y `LabelEncoder`.

**Parámetros:**
- `directory` — ruta destino; se crea recursivamente si no existe.

---

### Propiedad `if_model -> IsolationForestModel`

**Descripción:** Retorna el modelo de Estadio 1. Permite al código externo acceder al IF sin romper el encapsulamiento mediante atributos privados.

---

### Propiedad `stage2_classifier -> XGBoostClassifier`

**Descripción:** Retorna el clasificador de Estadio 2. Permite al código externo acceder al XGBoost (p. ej., para obtener `predict_proba` y `classes_`) sin acceder directamente a `_xgb`.

---

### `get_feature_importance() -> Dict[str, float]`

**Descripción:** Delega en `XGBoostClassifier.get_feature_importance()` y retorna la importancia de cada feature (tipo `weight`) ordenada de mayor a menor. Usado por el módulo de visualización para generar `feature_importance.png`.

---

### `get_if_operating_point() -> dict`

**Descripción:** Retorna el diccionario de diagnóstico del Estadio 1: umbral activo, $\beta$, $F_\beta$-score, precisión, recall, FPR, FNR y conteos de TP/FP/TN/FN. Usado para generar `pr_tradeoff.png` y `threshold_curve.png`.

## Dependencias Internas

- [`config`](../../config.py) — `Config`, `cfg`, `BENIGN_LABEL`.
- [`models.isolation_forest`](../../models/isolation_forest.py) — `IsolationForestModel`.
- [`models.xgboost_classifier`](../../models/xgboost_classifier.py) — `XGBoostClassifier`.
- [`pipelines.base`](../../pipelines/base.py) — `BasePipeline`.

## Referencias

- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of KDD 2016, 785–794.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation forest*. Proceedings of IEEE ICDM 2008, 413–422.
- Van Etten, J. *et al.* (2018). *CICIDS2017: A realistic cyber intrusion detection dataset*. IEEE Access.
