# Módulo `models.neural_network`

## Descripción General

Este módulo implementa el clasificador de Stage 2 del Pipeline B: un perceptrón multicapa (MLP) construido íntegramente con NumPy, sin dependencias de frameworks de deep learning. Recibe las muestras clasificadas como anómalas por el Isolation Forest (Stage 1) y predice el tipo de ataque entre las clases canónicas del esquema CICIDS2017. La implementación comprende normalización de entradas con `StandardScaler`, paso directo (*forward pass*), retropropagación completa mediante la regla de la cadena, Batch Normalisation en todas las capas ocultas, Dropout invertido, optimizador Adam con penalización L2, pérdida ponderada por clase, entropía cruzada con suavizado de etiquetas (*label smoothing*), parada temprana con restauración de los mejores pesos, y reducción automática de la tasa de aprendizaje (*ReduceLROnPlateau*).

---

## Fundamento Teórico

### 2.1 Arquitectura del Modelo

La red emplea la siguiente topología fija:

```
Input (n_features)
  →  StandardScaler  (ajustado sobre el train set)
  →  Dense(256)  →  BatchNorm  →  ReLU  →  Dropout(p = 0.15)
  →  Dense(128)  →  BatchNorm  →  ReLU  →  Dropout(p = 0.15)
  →  Dense(64)   →  BatchNorm  →  ReLU  →  Dropout(p = 0.10)
  →  Dense(n_classes)  →  Softmax
```

Las tres capas ocultas incluyen Batch Normalisation para acelerar la convergencia y eliminar la sensibilidad a la escala de las entradas, que en CICIDS2017 varía en varios órdenes de magnitud (bytes: 0–10⁷; duraciones: 0–10⁹). El `StandardScaler` se ajusta una única vez en `fit()` y se aplica en `predict()` y `predict_proba()`; se persiste junto a los pesos para garantizar coherencia en inferencia.

### 2.2 Inicialización He

Los pesos de cada capa densa se inicializan según la propuesta de He *et al.* (2015), diseñada para redes con activaciones ReLU:

$$W^{(l)} \sim \mathcal{N}\!\left(0,\;\sqrt{\dfrac{2}{n_{\text{in}}^{(l)}}}\right)$$

donde $n_{\text{in}}^{(l)}$ es el número de unidades en la capa anterior. Esta varianza preserva la magnitud de la señal en el paso directo y del gradiente en la retropropagación.

### 2.3 Batch Normalisation (Ioffe & Szegedy, 2015)

Dado un mini-lote $\mathbf{Z} \in \mathbb{R}^{N \times H}$, el paso directo en modo entrenamiento opera como:

$$\mu = \frac{1}{N}\sum_{i=1}^{N} z_i, \quad \sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(z_i - \mu)^2$$

$$\hat{z}_i = \frac{z_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \quad y_i = \gamma \odot \hat{z}_i + \beta$$

Los parámetros $\gamma$ (escala) y $\beta$ (desplazamiento) se optimizan junto con los pesos de la red. Las medias y varianzas móviles exponenciales:

$$\mu_{\text{run}} \leftarrow \rho\,\mu_{\text{run}} + (1-\rho)\,\mu, \quad \sigma^2_{\text{run}} \leftarrow \rho\,\sigma^2_{\text{run}} + (1-\rho)\,\sigma^2$$

con $\rho = 0.9$, se emplean durante la inferencia en sustitución de las estadísticas de lote.

**Gradiente del paso backward de BatchNorm.** Aplicando la regla de la cadena sobre la dependencia de $\mu$ y $\sigma^2$ respecto a todos los elementos del lote se obtiene la expresión cerrada:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{N\,\sigma}\left(N \frac{\partial \mathcal{L}}{\partial \hat{z}_i} - \sum_{j}\frac{\partial \mathcal{L}}{\partial \hat{z}_j} - \hat{z}_i\sum_{j}\frac{\partial \mathcal{L}}{\partial \hat{z}_j}\hat{z}_j\right)$$

con $\sigma = \sqrt{\sigma^2 + \varepsilon}$.

### 2.4 Función de Activación ReLU

$$\text{ReLU}(z) = \max(0, z)$$

Su derivada es la función escalón $\mathbf{1}[z > 0]$, lo que suprime el gradiente para activaciones negativas y lo propaga sin atenuación para las positivas.

### 2.5 Dropout Invertido

Durante el entrenamiento, cada unidad se suprime de forma independiente con probabilidad $p$. La variante invertida reescala las unidades conservadas para que la expectativa de la salida sea idéntica en entrenamiento e inferencia:

$$\text{mask}_j \sim \text{Bernoulli}(1-p), \quad a'_j = a_j \cdot \frac{\text{mask}_j}{1 - p}$$

En inferencia, la capa se omite sin modificar la señal. El gradiente en el paso backward es simplemente $\partial \mathcal{L}/\partial a_j = (\partial \mathcal{L}/\partial a'_j)\cdot\text{mask}_j/(1-p)$, que ya está absorbido en la máscara.

### 2.6 Entropía Cruzada con Suavizado de Etiquetas y Pesos de Clase

Para reducir la sobreconfianza del clasificador y mejorar la calibración, los vectores *one-hot* $\mathbf{y}$ se reemplazan por etiquetas suavizadas:

$$\tilde{y}_k = (1 - \varepsilon)\,\delta_{k,c} + \frac{\varepsilon}{K}$$

donde $c$ es la clase verdadera, $K$ el número de clases y $\varepsilon = 0.05$. La función de pérdida incorpora pesos de clase para compensar el desequilibrio extremo del dataset:

$$w_c = \frac{N}{K \cdot n_c}$$

donde $n_c$ es el número de muestras de la clase $c$ en el conjunto de entrenamiento. La pérdida ponderada es:

$$\mathcal{L} = -\frac{\sum_{i=1}^{N} w_{c_i}\sum_{k=1}^{K}\tilde{y}_{ik}\,\log p_{ik}}{\sum_{i=1}^{N} w_{c_i}}$$

Los pesos se calculan en `fit()` a partir del conteo de etiquetas codificadas y se aplican en cada paso *forward* del mini-lote mediante `sample_weights[y_b]`.

### 2.6 bis — Focal Loss

La pérdida estándar de entropía cruzada trata todas las muestras por igual, lo que resulta ineficiente cuando la mayoría de los ejemplos son clasificados correctamente con alta confianza: el gradiente queda dominado por muestras fáciles en lugar de centrarse en los casos difíciles o mal clasificados.

La **Focal Loss** (Lin et al., 2017) introduce un factor de modulación que reduce automáticamente la contribución de las muestras fáciles:

$$\text{FL}(p_t) = (1 - p_t)^\gamma \cdot \text{CE}(p_t)$$

donde $p_t$ es la probabilidad asignada a la clase verdadera y $\gamma \geq 0$ es el parámetro de enfoque (`focal_gamma = 2.0`). Cuando una muestra se clasifica correctamente con alta confianza ($p_t \to 1$), el factor $(1 - p_t)^\gamma \to 0$ atenúa su contribución; cuando se clasifica mal ($p_t \to 0$), el factor se aproxima a 1 y la pérdida equivale a la entropía cruzada estándar.

En la implementación, el factor focal se combina multiplicativamente con los pesos de clase para tratar simultáneamente el desequilibrio de clases y la dificultad de cada muestra:

```python
p_t     = probs[np.arange(len(y_b)), y_b]          # prob. de la clase verdadera
focal_w = (1.0 - p_t) ** config.focal_gamma         # factor focal por muestra
sample_w = focal_w * class_weights[y_b]             # pesos finales por muestra
```

El vector `sample_w` reemplaza a `class_weights[y_b]` como argumento de `_cross_entropy_loss` y de `_backward`, de modo que el gradiente queda ponderado por la dificultad de cada ejemplo además de por la frecuencia de su clase.

**Referencia:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017. arXiv:1708.02002.

### 2.7 Retropropagación

El gradiente conjunto de la capa Softmax y la entropía cruzada respecto a los logits pre-Softmax $\mathbf{z}^{(4)}$ es:

$$\frac{\partial \mathcal{L}}{\partial z^{(4)}_i} = \frac{1}{N}\left(p_i - \tilde{y}_i\right) \cdot \hat{w}_i$$

donde $\hat{w}_i = w_{c_i} / \bar{w}$ es el peso normalizado de la muestra $i$ (con $\bar{w}$ la media de los pesos del lote). Esta expresión evita la inestabilidad numérica de computar explícitamente el Jacobiano de Softmax. El gradiente se propaga hacia atrás mediante la regla de la cadena estándar para capas densas:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \mathbf{a}^{(l-1)\top} \cdot \delta^{(l)}, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \sum_i \delta^{(l)}_i, \quad
\delta^{(l-1)} = \delta^{(l)} \cdot \mathbf{W}^{(l)\top}$$

### 2.8 Optimizador Adam (Kingma & Ba, 2015)

Adam mantiene estimaciones del primer y segundo momento del gradiente con corrección de sesgo:

$$m_t = \beta_1\,m_{t-1} + (1-\beta_1)\,g_t$$
$$v_t = \beta_2\,v_{t-1} + (1-\beta_2)\,g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \alpha\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

Los valores por defecto son $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, $\alpha=10^{-3}$. La regularización L2 se implementa como decaimiento de pesos (*weight decay*) añadiendo $\lambda\mathbf{W}$ al gradiente antes de la actualización, con $\lambda = 10^{-4}$.

### 2.9 Parada Temprana, LR Scheduling y Gestión de Memoria

Al final de cada epoch se evalúa el F1-macro en el conjunto de validación (`es_metric = -val_f1_mac`). Si no mejora en más de $\delta_{\min} = 10^{-4}$ durante $p_{\text{es}} = 20$ epochs consecutivas, el entrenamiento se detiene y se restauran los pesos del epoch con mayor F1-macro de validación.

Adicionalmente, se implementa **ReduceLROnPlateau**: si el F1-macro de validación no mejora durante $p_{\text{lr}} = 5$ epochs consecutivas, la LR actual se multiplica por un factor $f = 0.5$, con un mínimo de $\text{lr}_{\min} = 10^{-5}$. El contador de paciencia del scheduler se reinicia tanto al mejorar el F1-macro como al ejecutar una reducción:

$$\text{lr}_{t+1} = \max\!\left(\text{lr}_t \times f,\;\text{lr}_{\min}\right) \quad \text{si } \text{counter}_{\text{lr}} \geq p_{\text{lr}}$$

Los dos contadores (early stopping y LR scheduler) son independientes pero comparten la condición de mejora: al mejorar el F1-macro de validación, ambos se reinician a 0. La LR actual se almacena en `self._current_lr` y se usa en `_adam_update()` en lugar del valor estático de configuración, por lo que el scheduler actúa automáticamente sobre todos los parámetros.

Inmediatamente tras restaurar los mejores pesos, la copia de seguridad (`_best_weights`) se libera asignándola a `None`, eliminando la duplicación de memoria.

#### LR Warmup

Durante las primeras `lr_warmup_epochs = 5` épocas la tasa de aprendizaje se escala linealmente desde `lr_min` hasta `learning_rate`, evitando actualizaciones de gran magnitud en los primeros pasos cuando los pesos aún están alejados de una región útil:

$$\text{lr}_\text{epoch} = \text{lr}_{\min} + (\text{lr}_{\text{init}} - \text{lr}_{\min}) \cdot \frac{\text{epoch}}{\texttt{lr\_warmup\_epochs}}$$

Una vez superado el periodo de calentamiento, el LR scheduler (ReduceLROnPlateau) toma el control.

#### Gradient Clipping

Antes de cada actualización Adam, si la norma L2 global de todos los gradientes supera `grad_clip_norm = 1.0`, se reescalan uniformemente de modo que la norma resultante sea exactamente `grad_clip_norm`:

$$\text{si } \|\mathbf{g}\|_2 > \texttt{grad\_clip\_norm}: \quad g_i \leftarrow g_i \cdot \frac{\texttt{grad\_clip\_norm}}{\|\mathbf{g}\|_2 + \varepsilon}$$

El clipping se aplica sobre el vector concatenado de todos los parámetros (pesos, sesgos y parámetros BatchNorm) antes de que Adam calcule sus estimaciones de momento. Poner `grad_clip_norm = 0.0` desactiva el mecanismo.

#### Balanced Batching

Cuando `use_balanced_batching = True`, en cada época los índices no se permutan aleatoriamente sino que se muestrean de forma estratificada por clase: se selecciona un número igual de ejemplos por clase (`per_class = batch_size // n_classes`) en cada mini-batch. Para clases con menos ejemplos disponibles que `per_class` el muestreo se hace con reemplazo. Esto garantiza que cada batch contenga señal de todas las clases, evitando épocas dominadas por la clase BENIGN (≈ 83 % de las muestras) que anulan la actualización del gradiente para clases minoritarias como Botnet.

---

## Funciones / Clases

### `NeuralNetwork(config, n_features, n_classes, random_seed)`

**Descripción:** Instancia la red neuronal inicializando los parámetros y el estado del optimizador. La arquitectura es fija conforme a la especificación del Pipeline B.

**Parámetros:**
- `config` — instancia de `NeuralNetworkConfig` con hiperparámetros
- `n_features` — dimensionalidad del vector de entrada (número de características tras preprocesado)
- `n_classes` — número de clases del problema (máximo 7 en el esquema CICIDS2017)
- `random_seed` — semilla para reproducibilidad (He init y máscaras Dropout)

---

### `fit(X_train, y_train, X_val, y_val) -> NeuralNetwork`

**Descripción:** Entrena la red con descenso de gradiente estocástico en mini-lotes, Adam y parada temprana. Internamente: (1) ajusta el `StandardScaler` sobre `X_train` y lo aplica a `X_val`; (2) calcula los pesos de clase inversos a la frecuencia si `use_class_weights=True`; (3) ejecuta el bucle de épocas con LR scheduling automático; (4) restaura los mejores pesos al finalizar. Antes de codificar `y_val`, filtra cualquier muestra cuya etiqueta no esté presente en el conjunto de entrenamiento, emitiendo un `WARNING` con el recuento y el conjunto de clases descartadas; esto garantiza que `LabelEncoder.transform` no lance `ValueError` cuando una clase de ataque rara no aparece en las anomalías de entrenamiento enrutadas por el Stage 1.

**Parámetros:**
- `X_train` — matriz de características del conjunto de entrenamiento, shape `(n_train, n_features)`
- `y_train` — etiquetas de clase en cadena de texto, shape `(n_train,)`
- `X_val` — matriz de características de validación, shape `(n_val, n_features)`
- `y_val` — etiquetas de validación en cadena de texto, shape `(n_val,)`

**Retorna:** `self`

**Efectos secundarios:** Rellena `self.history` con las curvas de pérdida y exactitud por epoch.

---

### `predict(X) -> np.ndarray`

**Descripción:** Devuelve las etiquetas de clase predichas para cada muestra.

**Parámetros:**
- `X` — shape `(n_samples, n_features)`

**Retorna:** `np.ndarray` de tipo `str`, shape `(n_samples,)` — etiquetas canónicas del esquema CICIDS2017.

---

### `predict_proba(X) -> np.ndarray`

**Descripción:** Devuelve la distribución de probabilidad sobre las clases.

**Parámetros:**
- `X` — shape `(n_samples, n_features)`

**Retorna:** `np.ndarray`, shape `(n_samples, n_classes)`. Las columnas siguen el orden de `self.classes_`.

---

### `save(directory) / load(directory)`

**Descripción:** Persistencia mediante `numpy.savez` para pesos y estadísticas BatchNorm, y `joblib` para el `LabelEncoder` y el `StandardScaler`.

**Archivos generados:**
- `nn_weights.npz` — todos los parámetros entrenables: `W1`, `b1`, `gamma1`, `beta_bn1`, `W2`, `b2`, `gamma2`, `beta_bn2`, `W3`, `b3`, `gamma3`, `beta_bn3`, `W4`, `b4`
- `nn_bn_running.npz` — medias y varianzas móviles de BatchNorm para las tres capas: `mean1`, `var1`, `mean2`, `var2`, `mean3`, `var3`
- `nn_label_encoder.joblib` — codificador de etiquetas
- `nn_scaler.joblib` — `StandardScaler` ajustado sobre el train set; necesario para reproducir la normalización en inferencia

---

### Métodos internos

| Método | Responsabilidad |
|--------|----------------|
| `_init_params()` | He init de pesos; BN params (γ, β) en identidad para las 3 capas ocultas |
| `_init_adam_state()` | Ceros para momentos `m`, `v`; `t = 0` |
| `_forward(X, training)` | Paso directo con caché para backprop; aplica BN en capas 1–3 |
| `_backward(cache, y_smooth, sample_weights)` | Retropropagación completa; propaga gradientes BN en capas 1–3; aplica pesos de muestra |
| `_adam_update(grads)` | Actualización Adam con corrección de sesgo; usa `self._current_lr` (modificable por el LR scheduler) |
| `_bn_forward(z, layer_id, training)` | BN forward genérico por `layer_id`; actualiza estadísticas móviles en entrenamiento |
| `_bn_backward(d_out, cache)` | BN backward con fórmula cerrada |
| `_dropout_forward(a, rate, training)` | Dropout invertido |
| `_dropout_backward(d_out, mask)` | Gradiente a través de Dropout |
| `_softmax(z)` | Softmax numéricamente estable |
| `_smooth_labels(y)` | Conversión a etiquetas suavizadas (ε = 0.05) |
| `_cross_entropy_loss(probs, y_smooth, sample_weights)` | Pérdida por entropía cruzada; soporta pesos por muestra |

---

## Dependencias Internas

- [`config.NeuralNetworkConfig`](../../config.py) — hiperparámetros de arquitectura y optimización

---

## Referencias

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *ICCV 2015*. arXiv:1502.01852.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*. arXiv:1502.03167.
- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*. arXiv:1412.6980.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15(56), 1929–1958.
- Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS 2019*. arXiv:1906.02629.
