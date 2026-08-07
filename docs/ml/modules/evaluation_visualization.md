# Módulo `evaluation.visualization`

## Descripción General

El módulo `evaluation/visualization.py` encapsula la generación de la totalidad de las figuras del proyecto CICIDS2017, organizadas en cuatro grupos funcionales: análisis exploratorio de datos (EDA), diagnóstico del Pipeline A (IF + XGBoost), diagnóstico del Pipeline B (IF + Red Neuronal NumPy) y comparativa entre ambos pipelines. Todas las figuras se persisten en disco mediante la función auxiliar `_save_fig`, con una resolución de 150 dpi y gestión automática de la creación de directorios. El módulo no produce ninguna salida en pantalla; emplea exclusivamente el backend no interactivo de matplotlib para garantizar la ejecución reproducible en entornos sin servidor gráfico.

---

## Fundamento Teórico

### Curva Precision-Recall y operación con F-beta

La curva Precision-Recall (PR) traza la relación entre precisión $P$ y exhaustividad $R$ a medida que varía el umbral de decisión. Para un clasificador binario con umbral $t$:

$$P(t) = \frac{TP(t)}{TP(t)+FP(t)}, \quad R(t) = \frac{TP(t)}{TP(t)+FN(t)}$$

El punto de operación óptimo se selecciona maximizando la métrica $F_\beta$:

$$F_\beta = \frac{(1+\beta^2)\,P\,R}{\beta^2\,P + R}$$

donde $\beta > 1$ penaliza los falsos negativos (prioriza exhaustividad) y $\beta < 1$ penaliza los falsos positivos (prioriza precisión). El contorno iso-$F_\beta$ que atraviesa el punto de operación $(P_0, R_0)$ con valor $F_0$ se obtiene despejando $P$ en función de $R$:

$$P = \frac{F_0 \, R}{(1+\beta^2)\,R - F_0\,\beta^2}$$

Esta curva parametriza todos los pares $(P, R)$ equivalentes en términos de $F_\beta$ e ilustra el coste de alejarse del punto óptimo.

### Curva ROC y AUC

La curva Receiver Operating Characteristic (ROC) representa la tasa de verdaderos positivos (TPR = Recall) frente a la tasa de falsos positivos (FPR) para un clasificador binario One-vs-Rest:

$$\text{TPR} = \frac{TP}{TP+FN}, \quad \text{FPR} = \frac{FP}{FP+TN}$$

El área bajo la curva ROC (AUC) cuantifica la capacidad discriminativa del modelo independientemente del umbral. Un clasificador aleatorio describe la diagonal $\text{TPR} = \text{FPR}$ con AUC = 0.5.

### Isolation Forest: inversión de scores para curvas sklearn

La función `decision_function` de sklearn devuelve valores donde los menores corresponden a las muestras más anómalas. Las funciones `precision_recall_curve` y `roc_curve` de sklearn esperan scores en el sentido contrario (mayor score = más positivo). Por ello, en `plot_pr_tradeoff` se invierte el signo: se pasa `-scores` a `precision_recall_curve`, de modo que las curvas resultantes son coherentes con el convenio de sklearn.

---

## Funciones

### `_save_fig(fig: plt.Figure, path: Path) -> None`
**Descripción:** Guarda *fig* en *path*, crea el directorio padre si no existe y cierra la figura para liberar memoria.  
**Parámetros:**
- `fig` — objeto `matplotlib.figure.Figure` a persistir
- `path` — ruta completa del archivo de salida (`.png`)

**Notas:** Siempre aplica `tight_layout()` y `bbox_inches="tight"` para evitar recortes en los ejes.

---

### `plot_class_distribution(y, output_dir, *, title) -> None`
**Descripción:** Genera un gráfico de barras con el recuento de muestras por clase canónica y lo guarda como `class_distribution.png`.  
**Parámetros:**
- `y` — etiquetas de clase, `pd.Series` o `np.ndarray` de strings, forma `(n,)`
- `output_dir` — directorio de destino (p. ej. `output/eda/`)
- `title` — título de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/class_distribution.png`  
**Interpretación:** Permite detectar desequilibrio severo entre clases antes de entrenar. Las anotaciones numéricas sobre cada barra facilitan la comparación exacta.

---

### `plot_feature_correlation(X, output_dir, *, top_n, title) -> None`
**Descripción:** Calcula la correlación de Pearson entre los `top_n` atributos de mayor varianza y la representa como mapa de calor, guardando el resultado como `feature_correlation.png`.  
**Parámetros:**
- `X` — matriz de características, `pd.DataFrame` o `np.ndarray`, forma `(n, p)`
- `output_dir` — directorio de destino
- `top_n` — número de atributos seleccionados por varianza descendente (default: 20)
- `title` — título de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/feature_correlation.png`  
**Interpretación:** Las regiones de alta correlación (> |0.8|) identifican grupos de atributos redundantes y fundamentan decisiones de reducción de dimensionalidad.

---

### `plot_feature_distributions(X, y, output_dir, *, top_n, title) -> None`
**Descripción:** Genera una cuadrícula de gráficos KDE de los `top_n` atributos de mayor varianza, con una curva por clase, y lo guarda como `feature_distributions.png`.  
**Parámetros:**
- `X` — matriz de características alineada con `y`
- `y` — etiquetas de clase, forma `(n,)`
- `output_dir` — directorio de destino
- `top_n` — número de atributos a visualizar (default: 6)
- `title` — supertítulo de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/feature_distributions.png`  
**Notas:** Si el estimador KDE falla (datos sin varianza o muy escasos), la función cae automáticamente a un histograma normalizado.  
**Interpretación:** Las distribuciones solapadas entre BENIGN y clases de ataque indican atributos de baja discriminabilidad; las distribuciones separadas señalan atributos informativos para Stage 1 (Isolation Forest).

---

### `plot_confusion_matrix(y_true, y_pred, labels, output_dir, *, title, normalize) -> None`
**Descripción:** Genera la matriz de confusión normalizada por filas y la guarda como `confusion_matrix.png` en el directorio indicado. Esta función es compartida por los diagnósticos de Pipeline A y Pipeline B.  
**Parámetros:**
- `y_true` — etiquetas verdaderas, forma `(n,)`
- `y_pred` — etiquetas predichas, forma `(n,)`
- `labels` — orden canónico de clases (filas/columnas)
- `output_dir` — directorio de destino (`output/pipeline_a/` o `output/pipeline_b/`)
- `title` — título de la figura
- `normalize` — modalidad de normalización: `'true'` (por fila), `'pred'` (por columna), `'all'`

**Retorna:** `None`  
**Figura generada:** `{output_dir}/confusion_matrix.png`  
**Interpretación:** La diagonal principal indica la tasa de acierto por clase (recall). Los errores fuera de la diagonal muestran las confusiones más frecuentes, p. ej. DoS clasificado como DDoS.

---

### `plot_feature_importance(feature_importance, output_dir, *, top_n, title) -> None`
**Descripción:** Representa las `top_n` importancias de atributos de XGBoost como gráfico de barras horizontales, ordenadas de mayor a menor, y guarda el resultado como `feature_importance.png`.  
**Parámetros:**
- `feature_importance` — `Dict[str, float]` ordenado descendentemente, tal como lo devuelve `XGBoostClassifier.get_feature_importance()`
- `output_dir` — directorio de destino (`output/pipeline_a/`)
- `top_n` — número de atributos a mostrar (default: 20)
- `title` — título de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/feature_importance.png`  
**Interpretación:** Identifica qué atributos del tráfico de red son más determinantes para la clasificación supervisada de ataques. Una importancia muy concentrada en pocos atributos puede indicar sobreajuste o dependencia de artefactos del dataset.

---

### `plot_threshold_curve(scores, y_binary, optimal_threshold, output_dir, *, beta, n_points, title) -> None`
**Descripción:** Genera la curva de Precisión, Recall y F-beta en función del umbral de decisión del Isolation Forest, marcando el umbral óptimo con una línea vertical, y guarda el resultado como `threshold_curve.png`.  
**Parámetros:**
- `scores` — salida de `IsolationForestModel._model.decision_function()`, forma `(n,)`; valores más bajos indican mayor anomalía
- `y_binary` — etiqueta binaria de ground truth (1 = ataque, 0 = benigno), forma `(n,)`
- `optimal_threshold` — umbral devuelto por `IsolationForestModel.optimize_threshold()`
- `output_dir` — directorio de destino (`output/pipeline_a/`)
- `beta` — parámetro beta utilizado en la optimización
- `n_points` — número de candidatos de umbral a evaluar (default: 300)
- `title` — título de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/threshold_curve.png`  
**Complejidad:** O(`n_points` · `n`)  
**Interpretación:** Permite verificar que el umbral seleccionado coincide con el máximo de F-beta y analizar la sensibilidad del detector al desplazamiento del umbral.

---

### `plot_pr_tradeoff(scores, y_binary, optimal_threshold, output_dir, *, beta, title) -> None`
**Descripción:** Representa la curva Precision-Recall completa del Isolation Forest (clasificador binario), con el punto de operación marcado y el contorno iso-F-beta que lo atraviesa, guardando el resultado como `pr_tradeoff.png`.  
**Parámetros:**
- `scores` — salida de `decision_function()`, forma `(n,)` (negada internamente antes de pasarla a `precision_recall_curve`)
- `y_binary` — ground truth binario (1 = ataque, 0 = benigno)
- `optimal_threshold` — umbral óptimo seleccionado
- `output_dir` — directorio de destino (`output/pipeline_a/`)
- `beta` — parámetro beta del punto de operación

**Retorna:** `None`  
**Figura generada:** `{output_dir}/pr_tradeoff.png`  
**Interpretación:** La posición del punto de operación respecto a la curva PR muestra si el detector se encuentra cerca del frente de Pareto. El contorno iso-F-beta evidencia el coste de incrementar recall a expensas de precisión, o viceversa, y guía la selección de beta.

---

### `plot_training_curves(history, output_dir, *, title) -> None`
**Descripción:** Genera un panel de dos gráficos (pérdida y accuracy) con curvas de entrenamiento y validación por época, marcando el epoch de parada temprana, y guarda el resultado como `training_curves.png`.  
**Parámetros:**
- `history` — diccionario `{'train_loss', 'val_loss', 'train_acc', 'val_acc'}` producido por `NeuralNetwork.history` tras `fit()`
- `output_dir` — directorio de destino (`output/pipeline_b/`)
- `title` — supertítulo de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/training_curves.png`  
**Interpretación:** La divergencia creciente entre la curva de entrenamiento y la de validación indica sobreajuste. La convergencia de ambas curvas sin divergencia señala un buen equilibrio bias-varianza. La línea vertical roja indica el mejor epoch (mínimo val_loss), confirmando que el early stopping y la restauración de pesos actuaron correctamente.

---

### `plot_metrics_bar(comparison_result, output_dir, *, title) -> None`
**Descripción:** Gráfico de barras agrupadas que compara accuracy, F1-macro y F1-weighted entre Pipeline A y Pipeline B, guardando el resultado como `metrics_bar.png`.  
**Parámetros:**
- `comparison_result` — diccionario devuelto por `evaluation.metrics.compare_pipelines()`
- `output_dir` — directorio de destino (`output/comparison/`)
- `title` — título de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/metrics_bar.png`  
**Interpretación:** Permite comparar de un vistazo el rendimiento global de ambos sistemas. Las anotaciones numéricas sobre las barras facilitan la comparación cuantitativa directa.

---

### `plot_f1_per_class(metrics_a, metrics_b, output_dir, *, title) -> None`
**Descripción:** Gráfico de barras agrupadas del F1 por clase para cada pipeline, guardando el resultado como `f1_per_class.png`.  
**Parámetros:**
- `metrics_a` — diccionario de métricas de Pipeline A (de `compute_metrics`)
- `metrics_b` — ídem para Pipeline B
- `output_dir` — directorio de destino (`output/comparison/`)
- `title` — título de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/f1_per_class.png`  
**Interpretación:** Revela si uno de los pipelines supera al otro de forma consistente en todas las clases o solo en algunas. Las clases minoritarias (p. ej. Botnet, WebAttack) son diagnósticamente relevantes para evaluar la robustez ante desequilibrio.

---

### `plot_roc_curves(y_true, y_score_a, y_score_b, labels, output_dir, *, title) -> None`
**Descripción:** Genera curvas ROC One-vs-Rest para cada clase canónica, en subgráficos independientes, comparando Pipeline A (línea continua) y Pipeline B (línea discontinua), y guarda el resultado como `roc_curves.png`.  
**Parámetros:**
- `y_true` — etiquetas verdaderas, forma `(n,)`
- `y_score_a` — matriz de probabilidades del Pipeline A, forma `(n, n_classes)`
- `y_score_b` — ídem para Pipeline B
- `labels` — orden canónico de clases, coincidente con las columnas de `y_score_*`
- `output_dir` — directorio de destino (`output/comparison/`)
- `title` — supertítulo de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/roc_curves.png`  
**Notas:** La transformación One-vs-Rest se realiza mediante `sklearn.preprocessing.LabelBinarizer`. Las excepciones en el cálculo de curvas individuales (p. ej. clases con una sola muestra) se capturan silenciosamente.  
**Interpretación:** Una curva alejada de la diagonal y cercana a la esquina superior izquierda indica alta discriminabilidad. Las diferencias entre Pipeline A y B por clase revelan en qué categorías de ataque cada clasificador tiene ventaja.

---

### `plot_pr_curves(y_true, y_score_a, y_score_b, labels, output_dir, *, title) -> None`
**Descripción:** Genera curvas Precision-Recall One-vs-Rest por clase, con línea de referencia de clasificador aleatorio (prevalencia de clase), comparando ambos pipelines, y guarda el resultado como `pr_curves.png`.  
**Parámetros:**
- `y_true` — etiquetas verdaderas, forma `(n,)`
- `y_score_a` — probabilidades del Pipeline A, forma `(n, n_classes)`
- `y_score_b` — ídem para Pipeline B
- `labels` — orden canónico de clases
- `output_dir` — directorio de destino (`output/comparison/`)
- `title` — supertítulo de la figura

**Retorna:** `None`  
**Figura generada:** `{output_dir}/pr_curves.png`  
**Notas:** La línea de referencia horizontal (prevalencia de clase $= P(\text{positivo})$) indica el nivel que alcanzaría un clasificador que predice siempre la clase positiva. La etiqueta "Baseline" aparece únicamente en el primer subgráfico para evitar duplicados en la leyenda global.  
**Interpretación:** Las curvas PR son especialmente informativas en datasets desequilibrados (donde la curva ROC puede resultar optimista). Una curva situada claramente sobre la línea de referencia confirma capacidad discriminativa real.

---

## Resumen de Figuras Generadas

| Figura | Directorio | Función |
|---|---|---|
| `class_distribution.png` | `output/eda/` | `plot_class_distribution` |
| `feature_correlation.png` | `output/eda/` | `plot_feature_correlation` |
| `feature_distributions.png` | `output/eda/` | `plot_feature_distributions` |
| `confusion_matrix.png` | `output/pipeline_a/` | `plot_confusion_matrix` |
| `feature_importance.png` | `output/pipeline_a/` | `plot_feature_importance` |
| `threshold_curve.png` | `output/pipeline_a/` | `plot_threshold_curve` |
| `pr_tradeoff.png` | `output/pipeline_a/` | `plot_pr_tradeoff` |
| `confusion_matrix.png` | `output/pipeline_b/` | `plot_confusion_matrix` |
| `training_curves.png` | `output/pipeline_b/` | `plot_training_curves` |
| `metrics_bar.png` | `output/comparison/` | `plot_metrics_bar` |
| `f1_per_class.png` | `output/comparison/` | `plot_f1_per_class` |
| `roc_curves.png` | `output/comparison/` | `plot_roc_curves` |
| `pr_curves.png` | `output/comparison/` | `plot_pr_curves` |

---

## Dependencias Internas

- `config` — no se importa directamente; los directorios de salida se reciben como parámetros `Path` desde `main.py` a través de `cfg.pipeline_a_path`, `cfg.pipeline_b_path`, `cfg.eda_path` y `cfg.comparison_path`.
- `evaluation.metrics` — `plot_metrics_bar` y `plot_f1_per_class` consumen el diccionario devuelto por `compare_pipelines` y `compute_metrics`.
- `models.isolation_forest` — `plot_threshold_curve` y `plot_pr_tradeoff` reciben los `scores` de `IsolationForestModel.decision_function` y el umbral de `optimize_threshold`.
- `models.neural_network` — `plot_training_curves` consume `NeuralNetwork.history`.
- `models.xgboost_classifier` — `plot_feature_importance` consume `XGBoostClassifier.get_feature_importance()`.

## Referencias

- Fawcett, T. (2006). *An introduction to ROC analysis*. Pattern Recognition Letters, 27(8), 861–874.
- Davis, J., & Goadrich, M. (2006). *The relationship between Precision-Recall and ROC curves*. ICML 2006.
- van Rijsbergen, C.J. (1979). *Information Retrieval* (2ª ed.). Butterworth-Heinemann. [F-beta score]
- Scikit-learn developers (2024). *sklearn.metrics documentation*. https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
