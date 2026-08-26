# Módulo `models/xgboost_classifier`

## Descripción General

Este módulo implementa el segundo estadio (*Stage 2*) del Pipeline A del sistema de detección de intrusiones. La clase `XGBoostClassifier` encapsula un clasificador supervisado multi-clase basado en el algoritmo *Extreme Gradient Boosting* (XGBoost), cuya función es identificar el tipo específico de ataque de red en aquellas muestras que el detector de anomalías de Stage 1 (Isolation Forest) ha marcado como anómalas. El módulo integra un mecanismo de búsqueda de hiperparámetros por muestreo aleatorio (`RandomizedSearchCV`) y *early stopping* durante el entrenamiento final sobre el conjunto de validación, ambos con corrección del desbalance de clases mediante pesos muestrales inversamente proporcionales a la frecuencia de clase.

---

## Fundamento Teórico

### Gradient Boosting y XGBoost

El Gradient Boosting (Friedman, 2001) construye un modelo aditivo de la forma:

$$F_M(\mathbf{x}) = \sum_{m=1}^{M} \eta \cdot h_m(\mathbf{x})$$

donde $h_m$ es un árbol de decisión débil (*weak learner*) ajustado en cada iteración $m$ sobre los residuos negativos del gradiente de la función de pérdida $\mathcal{L}$. XGBoost (Chen & Guestrin, 2016) extiende este marco incorporando una regularización explícita en el objetivo:

$$\mathcal{L}^{(m)} = \sum_{i=1}^{n} \ell\!\left(y_i,\, \hat{y}_i^{(m-1)} + h_m(\mathbf{x}_i)\right) + \Omega(h_m)$$

$$\Omega(h) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2$$

donde $T$ es el número de hojas del árbol, $\mathbf{w}$ el vector de valores en las hojas, $\gamma$ penaliza la complejidad estructural y $\lambda$ (L2) penaliza la magnitud de los pesos. Mediante una aproximación de Taylor de segundo orden, el valor óptimo en cada hoja $j$ se obtiene analíticamente:

$$w_j^* = -\frac{\sum_{i \in \mathcal{I}_j} g_i}{\sum_{i \in \mathcal{I}_j} h_i + \lambda}$$

siendo $g_i = \partial_{\hat{y}} \ell(y_i, \hat{y}_i)$ y $h_i = \partial^2_{\hat{y}} \ell(y_i, \hat{y}_i)$ el gradiente y la hessiana de la pérdida respectivamente.

### Clasificación Multi-clase: Softmax

Para el problema de $K$ clases (K = 6 tipos de ataque) se emplea la función objetivo `multi:softprob`, que ajusta $K$ clasificadores binarios en paralelo mediante un esquema *one-vs-rest* y aplica la normalización softmax sobre las predicciones en hoja:

$$P(y = k \mid \mathbf{x}) = \frac{e^{F_k(\mathbf{x})}}{\sum_{j=1}^{K} e^{F_j(\mathbf{x})}}$$

La función de pérdida minimizada es la entropía cruzada categórica (*mlogloss*):

$$\mathcal{L}_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbf{1}[y_i = k] \log P(y_i = k \mid \mathbf{x}_i)$$

### Corrección del Desbalance de Clases

El dataset CICIDS2017 presenta una distribución muy sesgada hacia la clase `BENIGN` y entre los propios tipos de ataque. Para mitigar este desequilibrio en el entrenamiento del clasificador de Stage 2 se aplica una ponderación muestral *balanceada* (equivalente a la estrategia `class_weight='balanced'` de scikit-learn):

$$w_c = \frac{1}{f_c \cdot K}, \qquad \tilde{w}_c = \frac{w_c}{\bar{w}}$$

donde $f_c = n_c / n$ es la frecuencia relativa de la clase $c$, $K$ el número de clases y $\bar{w}$ la media de pesos (normalización a media unitaria). Cada muestra recibe el peso $\tilde{w}_{y_i}$ de su clase, lo que reequilibra la contribución de clases minoritarias en la función de pérdida.

### Búsqueda de Hiperparámetros: RandomizedSearchCV

La búsqueda aleatoria (Bergstra & Bengio, 2012) muestrea $n_{\text{iter}}$ combinaciones del espacio de hiperparámetros en lugar de recorrerlo exhaustivamente, lo que resulta computacionalmente viable cuando el espacio es grande. El espacio de búsqueda definido en `XGBoostConfig` es:

| Hiperparámetro | Valores |
|----------------|---------|
| `n_estimators` | {200, 300, 500} |
| `max_depth` | {4, 6, 8} |
| `learning_rate` | {0.05, 0.1, 0.2} |
| `subsample` | {0.8, 1.0} |
| `colsample_bytree` | {0.8, 1.0} |

La métrica de selección es el F1-macro promediado sobre $k = 3$ *folds* estratificados (`StratifiedKFold`), lo que garantiza que la distribución de clases se preserve en cada partición.

### Early Stopping

El entrenamiento del modelo final se detiene anticipadamente si la métrica de validación (`mlogloss`) no mejora en `early_stopping_rounds` rondas consecutivas. Formalmente, sea $\mathcal{L}_{\text{val}}^{(m)}$ la pérdida de validación en la iteración $m$; el entrenamiento se detiene en la ronda $\hat{m}$ tal que:

$$\hat{m} = \min \{ m : \mathcal{L}_{\text{val}}^{(m+r)} \geq \mathcal{L}_{\text{val}}^{(m)} \;\forall\, r \in \{1, \ldots, R\} \}$$

donde $R$ = `early_stopping_rounds`. El modelo utiliza los pesos de la iteración $\hat{m}$ como solución final.

---

## Clases y Métodos

### `XGBoostClassifier`

Clase principal que encapsula el ciclo de vida completo del clasificador de Stage 2.

**Constructor: `__init__(config: XGBoostConfig, random_seed: int = 42) -> None`**

**Parámetros:**
- `config` — instancia de `XGBoostConfig` con hiperparámetros por defecto y espacio de búsqueda (ver `config.py`)
- `random_seed` — semilla de reproducibilidad global

---

### `tune_hyperparameters(X_train, y_train, *, train_index=None) -> dict`

**Descripción:** Ejecuta `RandomizedSearchCV` sobre el espacio de búsqueda definido en `config.search_space`. El `LabelEncoder` interno se inicializa en esta llamada si aún no ha sido ajustado.

**Parámetros:**
- `X_train : np.ndarray` — matriz de características de los ejemplos de entrenamiento anómalos, shape `(n_samples, n_features)`
- `y_train : np.ndarray` — etiquetas de tipo de ataque en formato string, shape `(n_samples,)`
- `train_index : TemporalIndex | None` — posición cronológica (`order_idx`) y bloque temporal de cada fila de `X_train`, construido con `ml.flow.loader.build_temporal_index` y **restringido con la misma máscara de anomalía** que produjo `X_train`

**Retorna:** `dict` — mapa de hiperparámetros a sus valores óptimos encontrados

**Particionado del CV:** con `train_index`, los folds son cronológicos **dentro de cada bloque `(capture_day, raw_label)`** y de ventana expansiva: con `cv_folds=K` cada bloque se corta en `K+1` trozos y el fold *k* entrena en los trozos `0..k-1` y valida en el *k* (ver `ml/flow/cv.py`). Sin él —o si los folds degeneran, p. ej. bloques demasiado pequeños para cortarse, o un fold cuyo train tendría una sola clase— se recurre a `StratifiedKFold(shuffle=True)`, dejando constancia en el log (`xgb_tuning_temporal_folds_degenerate`). Un `train_index` cuya longitud no coincida con `X_train` **lanza `ValueError`**: es un error de programación, no una condición de los datos.

**Scorer:** `make_scorer(f1_score, average="macro", labels=np.arange(n_classes), zero_division=0)`, no la cadena `"f1_macro"`. Ésta promedia sobre `union(y_true, y_pred)` *del fold*, de modo que con folds cronológicos —donde que una clase falte en un trozo de validación es rutinario— el conjunto promediado varía por fold y por candidato y los scores de la búsqueda dejan de ser comparables. Es el mismo defecto que el `f1_macro` titular calculado sin `labels=`, un nivel más abajo.

**Notas:** No aplica *early stopping* en el proceso de búsqueda; éste se aplica exclusivamente en la llamada posterior a `fit`. Si esta función no es llamada, `fit` utiliza los valores por defecto de `config`. Los `sample_weight` se calculan una sola vez sobre todo el conjunto y sklearn los trocea por fold: el peso es propiedad de la clase, no del fold.

**Medido** sobre los CSV reales (`--tune --sample-size 200000`, protocolo por defecto, 22 de agosto de 2026): la entrada de la búsqueda son las **67.789 anomalías** que Stage 1 enruta, y los tres folds quedan en `train_sizes=[16943, 33893, 50836]` / `val_sizes=[16950, 16943, 16953]` con las **siete clases en las dos mitades de los tres folds** — ningún fold descartado y ningún fallback. `best_params={'subsample': 1.0, 'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.2, 'colsample_bytree': 0.8}` con `f1_macro=0.8316` en CV; sobre test, el run pasa de 0.6467 a 0.6584 de F1-macro. Reloj de pared del run completo: 602 s contra 86 s con `--skip-tuning`.

**Eventos de log del fallback:** los emite el pipeline, no este módulo — `xgb_tuning_folds_random_by_protocol` (INFO) cuando el run es `--split random` y no hay afirmación temporal que hacer, y `xgb_tuning_temporal_folds_unavailable` (WARNING) cuando el protocolo es temporal pero no llegó índice. Los folds efectivos se registran aquí, en `xgb_tuning_temporal_folds`.

---

### `fit(X_train, y_train, X_val, y_val) -> XGBoostClassifier`

**Descripción:** Entrena el modelo final combinando los parámetros por defecto de `config` con los hiperparámetros óptimos (si `tune_hyperparameters` fue invocado previamente). Aplica *early stopping* monitorizando `mlogloss` en el conjunto de validación. Antes de codificar `y_val`, filtra cualquier muestra cuya etiqueta no esté presente en las anomalías de entrenamiento, emitiendo un `WARNING` con el recuento y conjunto de clases descartadas; esto garantiza que `LabelEncoder.transform` no lance `ValueError` cuando una clase de ataque rara no llega al Stage 2 durante el entrenamiento.

**Parámetros:**
- `X_train : np.ndarray` — características de entrenamiento, shape `(n_train, n_features)`
- `y_train : np.ndarray` — etiquetas string de entrenamiento, shape `(n_train,)`
- `X_val   : np.ndarray` — características de validación, shape `(n_val, n_features)`
- `y_val   : np.ndarray` — etiquetas string de validación, shape `(n_val,)`

**Retorna:** `self` (permite encadenamiento de métodos)

---

### `predict(X) -> np.ndarray`

**Descripción:** Aplica el modelo ajustado para predecir el tipo de ataque. Internamente invierte la codificación del `LabelEncoder` para devolver strings en el esquema canónico de 7 clases.

**Parámetros:**
- `X : np.ndarray` — matriz de características, shape `(n_samples, n_features)`

**Retorna:** `np.ndarray` de `str`, shape `(n_samples,)` — etiquetas de tipo de ataque predichas

---

### `predict_proba(X) -> np.ndarray`

**Descripción:** Devuelve la distribución de probabilidad softmax sobre todas las clases para cada muestra.

**Parámetros:**
- `X : np.ndarray` — matriz de características, shape `(n_samples, n_features)`

**Retorna:** `np.ndarray`, shape `(n_samples, n_classes)` — probabilidades por clase, columnas ordenadas según `self.classes_`

---

### `get_feature_importance(feature_names=None) -> dict`

**Descripción:** Devuelve las importancias de características (tipo *weight*: número de veces que cada característica aparece en una partición de árbol) ordenadas de mayor a menor.

**Parámetros:**
- `feature_names : list of str, optional` — nombres legibles de columnas; si es `None` se usan claves `'f0'`, `'f1'`, ...

**Retorna:** `dict` — `{nombre_feature: importancia}` ordenado descendentemente por importancia

---

### `classes_ : np.ndarray` (propiedad)

**Descripción:** Etiquetas de clase en formato string, ordenadas según la codificación interna del `LabelEncoder`. Disponible tras el primer `fit` o `tune_hyperparameters`.

---

### `save(directory: Path) -> None`

**Descripción:** Serializa el `XGBClassifier` y el `LabelEncoder` mediante `joblib`.

**Archivos generados:**
- `xgboost.joblib` — modelo XGBoost serializado
- `xgb_label_encoder.joblib` — codificador de etiquetas serializado

---

### `load(directory: Path) -> XGBoostClassifier`

**Descripción:** Restaura el modelo y el codificador desde los archivos generados por `save`. Retorna `self`.

---

## Dependencias Internas

- [`config`](../../config.py) — `XGBoostConfig`
- [`utils/logger`](../utils/logger.md) — configuración del sistema de logging

## Referencias

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD), pp. 785–794. ACM. https://doi.org/10.1145/2939672.2939785
- Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. *The Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Bergstra, J., & Bengio, Y. (2012). *Random search for hyper-parameter optimization*. *Journal of Machine Learning Research*, 13, 281–305.
- Scikit-learn developers (2024). `sklearn.model_selection.RandomizedSearchCV`. Scikit-learn 1.4 documentation.
- XGBoost developers (2024). *XGBoost Documentation — Python API*. https://xgboost.readthedocs.io
