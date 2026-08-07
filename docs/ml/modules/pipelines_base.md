# Módulo `pipelines.base`

## Descripción General

El módulo `pipelines/base.py` define la clase abstracta `BasePipeline`, que establece el contrato de interfaz común para todos los pipelines de detección de intrusiones implementados en el proyecto. Su función es garantizar que `PipelineA` (Isolation Forest + XGBoost) y `PipelineB` (Isolation Forest + Red Neuronal NumPy) sean intercambiables desde el punto de vista del orquestador principal, haciendo posible la comparación sistemática y reproducible entre ambas estrategias de clasificación.

El diseño sigue el patrón de diseño **Template Method** en su variante abstracta pura: la clase base define la firma y la semántica de cada operación, mientras las subclases proveen la implementación concreta sin alterar la interfaz observable. Este enfoque garantiza la invarianza del protocolo experimental independientemente del clasificador de Stage 2 empleado.

## Fundamento Teórico

La arquitectura de los pipelines implementados sigue un esquema de **clasificación en cascada de dos etapas** (*two-stage cascade classifier*), formalizado como:

$$\hat{y}(x) = \begin{cases} \text{BENIGN} & \text{si } f_{\text{IF}}(x) \geq \tau^* \\ g(x) & \text{si } f_{\text{IF}}(x) < \tau^* \end{cases}$$

donde $f_{\text{IF}}(x)$ es la función de decisión del Isolation Forest (Stage 1), $\tau^*$ es el umbral óptimo obtenido maximizando el F-$\beta$ score sobre el conjunto de validación, y $g(x)$ es el clasificador de Stage 2 (XGBoost o MLP NumPy) que asigna la categoría de ataque específica.

La interfaz abstracta encapsula este flujo en cuatro operaciones: entrenamiento secuencial (`train`), inferencia en cascada (`predict`), evaluación cuantitativa (`evaluate`) y persistencia (`save`), asegurando que cualquier implementación cumpla el contrato necesario para el ciclo experimental.

## Funciones / Clases

### Clase `BasePipeline(ABC)`

**Descripción:** Clase base abstracta que define el contrato de todos los pipelines. Hereda de `abc.ABC`; no puede ser instanciada directamente.

---

#### `train(X_train, y_train, X_val, y_val, *, beta, skip_tuning) -> BasePipeline`

**Descripción:** Entrena secuencialmente el Stage-1 (Isolation Forest sobre muestras BENIGN) y el Stage-2 (clasificador de tipo de ataque). Optimiza el umbral del IF sobre el conjunto de validación usando F-$\beta$.

**Parámetros:**
- `X_train` — matriz de características de entrenamiento, shape `(n_train, n_features)`
- `y_train` — etiquetas de clase canónicas (string), shape `(n_train,)`
- `X_val` — matriz de características de validación, shape `(n_val, n_features)`
- `y_val` — etiquetas de clase canónicas (string), shape `(n_val,)`
- `beta` — peso F-$\beta$ para la optimización del umbral del IF. `beta > 1` penaliza falsos negativos; `beta < 1` penaliza falsos positivos. Valor por defecto: `1.0`
- `skip_tuning` — si `True`, omite la búsqueda de hiperparámetros del Stage-2 y entrena con los valores por defecto de `config.py`

**Retorna:** `self` (permite encadenamiento fluido)

---

#### `predict(X) -> np.ndarray`

**Descripción:** Ejecuta la inferencia en cascada completa. El Stage-1 clasifica cada muestra como BENIGN o anómala; las muestras anómalas se reenvían al Stage-2 para determinar el tipo de ataque. Las muestras BENIGN reciben directamente la etiqueta `'BENIGN'`.

**Parámetros:**
- `X` — matriz de características de entrada, shape `(n_samples, n_features)`

**Retorna:** `np.ndarray` de strings, shape `(n_samples,)`. Etiquetas de clase canónicas predichas para cada muestra.

**Notas:** La separación en dos etapas implica que el Stage-2 sólo procesa el subconjunto de muestras marcadas como anómalas por el Stage-1; las predicciones finales se reensamblan en el orden original de las muestras de entrada.

---

#### `evaluate(X_test, y_test) -> Dict[str, float]`

**Descripción:** Calcula métricas de clasificación extremo a extremo (*end-to-end*) sobre el conjunto de test. Internamente invoca `predict()` y compara las predicciones con las etiquetas reales. El diccionario devuelto debe incluir al menos las claves `'accuracy'`, `'f1_macro'` y `'f1_weighted'`; las implementaciones concretas pueden añadir métricas adicionales (p. ej., AUC-ROC por clase, métricas del Stage-1).

**Parámetros:**
- `X_test` — matriz de características del conjunto de test, shape `(n_test, n_features)`
- `y_test` — etiquetas de clase canónicas reales, shape `(n_test,)`

**Retorna:** `dict` con claves de tipo string y valores escalares float.

---

#### `save(directory) -> None`

**Descripción:** Serializa todos los componentes entrenados del pipeline (Stage-1, Stage-2, codificadores de etiquetas, umbrales) en el directorio especificado. El directorio se crea recursivamente si no existe. La implementación concreta debe garantizar que el pipeline pueda restaurarse completamente para inferencia sin necesidad de reentrenamiento.

**Parámetros:**
- `directory` — `Path` del directorio de destino

---

#### Propiedad `name -> str` (abstracta)

**Descripción:** Identificador corto legible por humanos empleado en mensajes de log y rutas de salida. Ejemplos de valores esperados: `'pipeline_a'`, `'pipeline_b'`.

---

#### `_log_evaluation(metrics) -> None` (método concreto)

**Descripción:** Método auxiliar concreto disponible para todas las subclases. Emite un mensaje `INFO` estructurado con el contenido del diccionario de métricas, formateado con cuatro decimales por valor. Facilita la uniformidad de los registros experimentales entre pipelines.

**Parámetros:**
- `metrics` — diccionario de métricas devuelto por `evaluate()`

## Dependencias Internas

- `config.py` — no importado directamente; los constructores de las subclases acceden a los dataclasses de configuración.
- `models/isolation_forest.py` — `IsolationForestModel`, empleado por todas las subclases como Stage-1.
- `models/xgboost_classifier.py` — `XGBoostClassifier`, Stage-2 de `PipelineA`.
- `models/neural_network.py` — `NeuralNetwork`, Stage-2 de `PipelineB`.

## Referencias

- Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. [Patrón Template Method, pp. 325–330]
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE International Conference on Data Mining*, 413–422.
- van Rossum, G., & Warsaw, B. (2001). PEP 3119 — Introducing Abstract Base Classes. Python Enhancement Proposals.
