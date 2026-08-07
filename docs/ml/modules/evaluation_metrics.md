# Módulo `evaluation.metrics`

## Descripción General

El módulo `evaluation/metrics.py` centraliza el cómputo estandarizado de métricas de clasificación multi-clase y la comparación cuantitativa entre los dos pipelines en cascada del proyecto (Pipeline A: IF + XGBoost; Pipeline B: IF + Red Neuronal NumPy). Provee además integración con `StratifiedKFold` de scikit-learn para obtener estimaciones robustas de la varianza del rendimiento sin necesidad de reentrenamiento. El módulo opera exclusivamente sobre predicciones y probabilidades de salida, manteniéndose independiente de la implementación interna de cada pipeline.

## Fundamento Teórico

### Accuracy

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N}$$

Fracción de instancias correctamente clasificadas sobre el total $N$. Es una métrica adecuada cuando las clases están balanceadas, pero engañosa en conjuntos con distribución sesgada (caso habitual en detección de intrusiones, donde BENIGN domina).

### F1-Score (macro y ponderado)

El F1-score de una clase $c$ es la media armónica de precisión y recall:

$$F_1^{(c)} = \frac{2 \cdot P^{(c)} \cdot R^{(c)}}{P^{(c)} + R^{(c)}}$$

donde $P^{(c)} = \text{TP}^{(c)} / (\text{TP}^{(c)} + \text{FP}^{(c)})$ y $R^{(c)} = \text{TP}^{(c)} / (\text{TP}^{(c)} + \text{FN}^{(c)})$.

- **F1 macro**: promedio no ponderado sobre todas las clases $\bar{F}_1 = \frac{1}{K}\sum_{c=1}^{K} F_1^{(c)}$. Penaliza igualmente el mal rendimiento en clases minoritarias, lo que lo hace relevante para la detección de ataques poco frecuentes (Botnet, WebAttack).
- **F1 ponderado**: $\tilde{F}_1 = \sum_{c=1}^{K} w_c F_1^{(c)}$ donde $w_c = n_c / N$. Refleja el rendimiento global ajustado por el soporte de cada clase.

### AUC-ROC (One-vs-Rest)

Para cada clase $c$, se construye un clasificador binario en esquema One-vs-Rest (OvR) y se calcula el área bajo la curva ROC. El eje $x$ representa la tasa de falsos positivos $\text{FPR} = \text{FP} / (\text{FP} + \text{TN})$ y el eje $y$ la tasa de verdaderos positivos (recall). Un clasificador aleatorio obtiene AUC = 0.5; un clasificador perfecto, AUC = 1.0. El **AUC macro** promedia los AUC individuales, siendo insensible al desequilibrio de clases.

### Validación Cruzada Estratificada (StratifiedKFold)

Dado un conjunto de datos $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, la validación cruzada con $K$ particiones estratificadas genera $K$ subconjuntos $\mathcal{D}_1, \ldots, \mathcal{D}_K$ tal que la distribución de clases se preserva en cada partición. Se evalúa el pipeline ya entrenado sobre cada $\mathcal{D}_k$, obteniéndose $K$ estimaciones de cada métrica. El módulo reporta la media $\bar{m}$ y la desviación estándar con corrección de Bessel:

$$\hat{\sigma} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (m_k - \bar{m})^2}$$

Esta estrategia estima la **varianza de la métrica** sobre la distribución del conjunto de test sin reentrenar los modelos.

## Funciones / Clases

### `compute_metrics(y_true, y_pred, y_score=None, labels=None) -> Dict[str, Any]`

**Descripción:** Calcula el conjunto completo de métricas de clasificación para un único experimento de predicción.

**Parámetros:**
- `y_true` — array de etiquetas cadena reales, shape `(n,)`
- `y_pred` — array de etiquetas cadena predichas, shape `(n,)`
- `y_score` — matriz de probabilidades softmax, shape `(n, n_classes)`; si se omite, no se calculan métricas AUC
- `labels` — lista ordenada de etiquetas canónicas; por defecto, unión ordenada de `y_true` y `y_pred`

**Retorna:** `dict` con claves `'accuracy'`, `'f1_macro'`, `'f1_weighted'`, `'f1_per_class'` (dict), y opcionalmente `'auc_roc_macro'` y `'auc_roc_per_class'` (dict).

**Notas:** La codificación OvR para AUC-ROC se realiza con `LabelBinarizer`; clases con un único valor en `y_true` producen AUC = NaN y se excluyen del promedio macro.

---

### `compare_pipelines(pipeline_a, pipeline_b, X_test, y_test, *, n_cv_splits=0, seed=42, labels=None, y_score_a=None, y_score_b=None) -> Dict[str, Any]`

**Descripción:** Evalúa y compara dos pipelines ya entrenados sobre el mismo conjunto de test. Opcionalmente, aplica `StratifiedKFold` para obtener estimaciones de varianza de las métricas.

**Parámetros:**
- `pipeline_a`, `pipeline_b` — objetos con interfaz `predict(X) -> np.ndarray` (compatible con `BasePipeline`)
- `X_test` — matriz de características de test, shape `(n_test, n_features)`
- `y_test` — etiquetas reales, shape `(n_test,)`
- `n_cv_splits` — número de pliegues para `StratifiedKFold`; 0 o 1 deshabilita la CV
- `seed` — semilla para `StratifiedKFold`
- `labels` — orden canónico de clases
- `y_score_a` — matriz de probabilidades softmax de Pipeline A, shape `(n_test, n_classes)`; cuando se proporciona, los resultados de Pipeline A incluyen `auc_roc_macro` y `auc_roc_per_class`
- `y_score_b` — ídem para Pipeline B

**Retorna:** `dict` con claves `'pipeline_a'`, `'pipeline_b'` (métricas completas, incluyendo AUC-ROC si se pasan las matrices de probabilidad), `'comparison'` (tabla de ganador y delta por métrica escalar más comparación per-clase de F1), y opcionalmente `'cv_pipeline_a'`/`'cv_pipeline_b'` (media/std por pliegue).

**Notas:** Los pipelines no se reentrenan durante la CV; la varianza estimada refleja la sensibilidad del rendimiento a la composición del conjunto de evaluación.

---

### `_eval_pipeline(pipeline, X, y, labels, y_score=None) -> Dict[str, Any]` *(privada)*

**Descripción:** Llama a `pipeline.predict(X)` y delega a `compute_metrics`. El parámetro opcional `y_score` se pasa directamente a `compute_metrics` para habilitar el cómputo de AUC-ROC cuando está disponible.

---

### `_compute_auc_roc(y_true, y_score) -> Dict[str, Any]` *(privada)*

**Descripción:** Binariza `y_true` con `LabelBinarizer` y calcula AUC-ROC por clase en esquema OvR. Retorna `'auc_roc_per_class'` y `'auc_roc_macro'`.

---

### `_cv_metric_stats(pipeline, X, y, n_splits, seed, labels) -> Dict[str, Dict[str, float]]` *(privada)*

**Descripción:** Ejecuta evaluación sobre $K$ pliegues estratificados de `(X, y)` sin reentrenamiento. Retorna media y desviación estándar (corrección de Bessel) de `accuracy`, `f1_macro` y `f1_weighted`.

**Complejidad:** O(K · n · forward_pass), donde `forward_pass` es el coste de inferencia del pipeline.

---

### `_build_comparison(metrics_a, metrics_b) -> Dict[str, Dict[str, Any]]` *(privada)*

**Descripción:** Genera una tabla con ganador y delta $\Delta = m_B - m_A$ para cada métrica escalar (`accuracy`, `f1_macro`, `f1_weighted`) y una comparación per-clase de F1 bajo la clave `'f1_per_class'`. Para cada clase $c$, el dict incluye `pipeline_a`, `pipeline_b` y `delta` (positivo indica superioridad de B). Un $\Delta > 0$ indica superioridad del Pipeline B.

---

### `_log_comparison(comparison) -> None` *(privada)*

**Descripción:** Emite un resumen estructurado de la comparativa mediante `logging.INFO`.

## Dependencias Internas

- `config` — importado indirectamente a través de los pipelines para conocer `CLASS_ORDER`
- `pipelines.base.BasePipeline` — interfaz que deben satisfacer los objetos pasados a `compare_pipelines`

## Referencias

- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.
- Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437.
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*, 14(2), 1137–1145.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
