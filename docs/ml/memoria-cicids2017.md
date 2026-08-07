---
title: Detección de Intrusiones de Red con Pipelines en Cascada sobre CICIDS2017
author: Guillermo Sanchez Perez
date: Junio 2026
Asignatura: Aprendizaje automatico
---

# Detección de Intrusiones de Red con Pipelines en Cascada sobre CICIDS2017

###### Guillermo Sánchez Pérez  -  Aprendizaje Automatico y Big Data

---

## 1. Introducción y descripción del problema

### 1.1 Contexto: sistemas de detección de intrusiones

La detección de intrusiones de red constituye uno de los pilares fundamentales de la ciberseguridad moderna. Un sistema de detección de intrusiones (*Intrusion Detection System*, IDS) tiene como objetivo identificar, en tiempo real o en diferido, patrones de tráfico de red que corresponden a actividad maliciosa, diferenciándolos del tráfico legítimo. Con el incremento continuo en el volumen y la sofisticación de los ciberataques, los enfoques basados en aprendizaje automático han desplazado progresivamente a los sistemas basados en firmas estáticas, que resultan incapaces de detectar variantes nuevas de ataques conocidos o amenazas no catalogadas.

La literatura especializada distingue dos grandes paradigmas en los IDS basados en ML: los métodos basados en la detección de anomalías, que identifican desviaciones respecto a un perfil de comportamiento normal aprendido, y los métodos de clasificación supervisada, que asignan cada flujo de tráfico a una categoría de ataque específica. Ambos enfoques presentan ventajas e inconvenientes complementarios: los detectores de anomalías generalizan bien a ataques desconocidos pero generan muchas falsas alarmas; los clasificadores supervisados ofrecen alta precisión en categorías conocidas pero dependen de la representatividad de sus datos de entrenamiento.

### 1.2 El dataset CICIDS2017 como benchmark estándar

El *Canadian Institute for Cybersecurity Intrusion Detection System 2017* (CICIDS2017) es el conjunto de datos de referencia más utilizado en la comunidad de IDS basados en ML. Fue generado en 2017 por la Universidad de New Brunswick mediante la simulación controlada de tráfico real en un entorno de red corporativa durante cinco días laborables, combinando flujos de tráfico legítimo con escenarios de ataque reproducidos de forma sistemática. Cada flujo de red fue capturado y caracterizado mediante la herramienta CICFlowMeter, que extrae 78 features estadísticas bidireccionales: longitudes de paquetes, tiempos entre llegadas (*inter-arrival times*), cuentas de flags TCP, ventanas de transmisión, tasas de transferencia y métricas de duración del flujo, entre otras.

La relevancia de CICIDS2017 como benchmark reside en varios factores: su gran volumen (más de 2,8 millones de flujos), la variedad de escenarios de ataque cubiertos (DoS, DDoS, escaneo de puertos, ataques de fuerza bruta, ataques web, botnet), y el hecho de que los datos proceden de una captura real y no de simulaciones sintéticas puras. No obstante, presenta también limitaciones bien documentadas, como la presencia de valores Inf/NaN debidos a divisiones por cero en el cálculo de tasas, un elevado porcentaje de filas duplicadas, y un desbalanceo de clases muy pronunciado que constituye el principal desafío técnico del proyecto.

### 1.3 Motivación de la arquitectura en cascada

El presente proyecto adopta una arquitectura en cascada de dos etapas (*two-stage cascade pipeline*) como estrategia central. El razonamiento que motiva este diseño es el siguiente:

En primer lugar, el desbalanceo extremo entre tráfico benigno y ataques hace que un clasificador multi-clase entrenado directamente sobre todos los datos tienda a sobreajustarse hacia la clase mayoritaria (BENIGN), que representa más del 58% de los datos. El modelo aprende que clasificar todo como tráfico normal ya produce una accuracy alta, con el consecuente colapso del rendimiento en clases minoritarias como Botnet o WebAttack.

En segundo lugar, la naturaleza del problema se presta conceptualmente a una separación en dos tareas: la primera es binaria y conceptualmente sencilla (*¿es este flujo normal o anómalo?*); la segunda es finamente discriminativa y requiere información de clase específica (*¿qué tipo de ataque es?*). Esta separación permite que cada etapa utilice el modelo óptimo para su tarea: un detector de anomalías no supervisado en Stage 1 y un clasificador supervisado de alta capacidad en Stage 2.

De este modo, el Stage 1 actúa como un filtro grueso que descarta con alta confianza el tráfico benigno, reduciendo drásticamente el espacio del problema para el Stage 2, que solo procesa los flujos identificados como anómalos. La arquitectura en cascada mejora así tanto la eficiencia computacional como la capacidad discriminativa del sistema en las clases de ataque minoritarias **(véase §9.4 para una cuantificación del grado real de reducción logrado)**.

### 1.4 Desafíos técnicos del problema

El problema plantea tres desafíos principales que condicionan todas las decisiones de diseño posteriores:

**Desbalanceo extremo de clases.** La distribución de clases en CICIDS2017 no es simplemente desbalanceada: es severamente asimétrica. Clases como Botnet (0,07%) o WebAttack (0,08%) tienen órdenes de magnitud menos representación que BENIGN (58,2%) o DoS (8,9%). Este desequilibrio implica que las métricas de evaluación estándar como la accuracy global resultan engañosas y que las técnicas de entrenamiento deben compensar activamente la subrepresentación de clases minoritarias.

**Alta dimensionalidad relativa.** Con 78 features numéricas y una muestra de trabajo de 1.000.001 filas, el ratio dimensionalidad/muestras no es crítico en términos absolutos, pero sí lo es por clases: una clase con 200 muestras en entrenamiento distribuidas sobre 78 features genera fronteras de decisión muy inestables sin regularización adecuada.

**Heterogeneidad semántica de los ataques.** Los distintos tipos de ataque difieren radicalmente en su mecanismo de red: un DoS satura el servidor con peticiones, un DDoS lo hace con tráfico distribuido, un escaneo de puertos realiza conexiones cortas y rápidas, y un ataque de fuerza bruta genera intentos de autenticación repetidos. Esta heterogeneidad implica que no existe un único conjunto de features discriminante para todos los tipos de ataque, lo que aumenta la dificultad del problema de clasificación multi-clase.

---

## 2. Objetivos del proyecto

### 2.1 Objetivo principal

El objetivo principal del proyecto es **diseñar, implementar y comparar experimentalmente dos pipelines de detección de intrusiones en cascada** sobre el dataset CICIDS2017, cuantificando el compromiso (*trade-off*) entre rendimiento predictivo y complejidad del clasificador de Stage 2:

- **Pipeline A**: Isolation Forest (Stage 1) + XGBoost (Stage 2)
- **Pipeline B**: Isolation Forest (Stage 1) + Red Neuronal implementada íntegramente en NumPy (Stage 2)

Ambos pipelines comparten el mismo detector de anomalías en Stage 1 y se diferencian únicamente en el clasificador multi-clase de Stage 2, lo que permite una comparación controlada entre un método de potenciación por gradiente (*gradient boosting*) altamente optimizado (XGBoost) y una red neuronal profunda implementada sin el uso de bibliotecas de aprendizaje profundo (*deep learning*).

### 2.2 Objetivos secundarios

**Implementación de una red neuronal completa en NumPy.** El Pipeline B exige la implementación manual del ciclo de entrenamiento completo: *forward pass*, *backpropagation*, actualización de pesos con Adam, normalización por lotes (*Batch Normalization*) con estadísticas diferenciadas en modo entrenamiento e inferencia, *Dropout* invertido, *Focal Loss*, y planificación de la tasa de aprendizaje (*learning rate scheduling*). Este objetivo tiene valor formativo en sí mismo, ya que obliga a comprender los fundamentos matemáticos de las redes neuronales sin la abstracción que ofrecen PyTorch o TensorFlow.

**Reproducibilidad total del experimento.** Todos los resultados deben ser reproducibles fijando una semilla global (`seed=42`) que controla NumPy, scikit-learn, XGBoost y el módulo `random` de Python. El guardado y carga de modelos (`--load-model`) permite recuperar resultados sin reentrenar.

**Análisis crítico de resultados.** Más allá de reportar métricas globales, el proyecto busca entender *por qué* cada pipeline se comporta de un modo específico en cada clase de ataque, examinando la importancia de features en XGBoost, las curvas de entrenamiento de la NN, y el impacto del detector de anomalías compartido en la cadena de errores.

---

## 3. Descripción y análisis del conjunto de datos

### 3.1 Origen y composición

El dataset CICIDS2017 fue publicado por el *Canadian Institute for Cybersecurity* de la Universidad de New Brunswick. Su generación requirió la construcción de una red corporativa simulada con perfiles de usuario realistas y la ejecución sistemática de distintos escenarios de ataque durante cinco días laborables de julio de 2017. El tráfico fue capturado en formato PCAP y procesado con CICFlowMeter para extraer features estadísticas a nivel de flujo bidireccional.

La distribución original del dataset consta de **8 archivos CSV**, uno por día y escenario de captura:

| Archivo CSV | Escenario principal |
|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | Únicamente tráfico BENIGN |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Ataques de fuerza bruta (FTP-Patator, SSH-Patator) |
| `Wednesday-workingHours.pcap_ISCX.csv` | DoS/DDoS (Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed) |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Ataques web (Brute Force, XSS, SQL Injection) |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Infiltración (escaneo + Dropbox) |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Botnet (ARES) + tráfico normal |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | DDoS (LOIT) |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Escaneo de puertos |

El dataset combinado contiene **2.830.743 filas** y **79 columnas**: 78 features numéricas de flujo de red y 1 columna de etiqueta (`Label`). Todos los archivos se cargan con encoding `latin-1`, necesario por las particularidades de codificación de los CSVs originales (véase §3.4).

### 3.2 Estructura de las features

Las 78 features son estadísticas computadas por CICFlowMeter a nivel de flujo bidireccional (Forward: origen→destino; Backward: destino→origen). Se pueden agrupar conceptualmente en las siguientes categorías:

| Categoría | Ejemplos de features | Relevancia IDS |
|---|---|---|
| Duración y tamaño del flujo | `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets` | Discrimina DoS (flujos largos) de escaneos (flujos cortos) |
| Longitudes de paquetes | `Fwd Packet Length Mean`, `Bwd Packet Length Std`, `Min Packet Length` | Distingue HTTP (payloads grandes) de SYN floods (payloads nulos) |
| Tiempos entre llegadas (IAT) | `Flow IAT Mean`, `Fwd IAT Std`, `Bwd IAT Min/Max` | Ataques automatizados tienen IAT muy regulares (baja varianza) |
| Flags TCP | `SYN Flag Count`, `RST Flag Count`, `PSH Flag Count` | Escaneos de puertos muestran RST elevado; SYN floods muestran SYN sin ACK |
| Tasas de transferencia | `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s` | DoS genera tasas anormalmente altas |
| Ventanas TCP | `Init_Win_bytes_forward`, `Init_Win_bytes_backward` | Indicativos de negociación de conexión |
| Features de actividad | `Active Mean/Std/Max`, `Idle Mean/Std/Max` | Distingue flujos continuos de flujos intermitentes |

Todas las features son numéricas continuas. No existen features categóricas ni texto libre, lo que simplifica el preprocesado pero amplifica el problema del desbalanceo al no disponer de representaciones simbólicas de los tipos de ataque.

### 3.3 Distribución de clases y motivación de la agrupación

#### 3.3.1 Etiquetas originales y taxonomía canónica

El dataset CICIDS2017 contiene **15 etiquetas de ataque originales** más la clase BENIGN. Muchas de estas etiquetas corresponden a variantes específicas de herramienta o técnica dentro de una misma familia de ataque. El proyecto agrupa estas etiquetas en **7 clases canónicas** mediante el diccionario `ATTACK_GROUPS` de `config.py`:

| Etiqueta original | Clase canónica | Fundamento de la agrupación |
|---|---|---|
| BENIGN | BENIGN | Tráfico legítimo, clase de referencia |
| DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed | DoS | Todos explotan el agotamiento de recursos del servidor mediante distintas técnicas HTTP/SSL; la feature discriminante es el comportamiento de flujo, no la herramienta |
| DDoS | DDoS | Ataque de denegación de servicio distribuido con características de flujo distintas al DoS unidireccional |
| PortScan, Infiltration | Scan | Ambos implican sondeo sistemático de hosts/puertos; Infiltration comienza con reconocimiento de red |
| FTP-Patator, SSH-Patator | BruteForce | Ataques de autenticación por fuerza bruta sobre distintos protocolos; el patrón de flujo (conexiones repetidas con credenciales) es idéntico |
| Web Attack – Brute Force, Web Attack – XSS, Web Attack – Sql Injection | WebAttack | Ataques en capa de aplicación HTTP; a nivel de flujo de red las diferencias entre variantes son mínimas |
| Bot | Botnet | Tráfico de comando y control (C2) de malware bot |

**Justificación de la agrupación.** La decisión de agrupar variantes en familias responde a varios criterios:

1. **Generalización semántica**: Un IDS que aprende a detectar "DoS Hulk" como clase independiente no generalizará a una nueva herramienta DoS no vista durante el entrenamiento. Al aprender el patrón genérico "DoS" (flujos de larga duración, alta tasa de paquetes, payloads repetitivos), el modelo captura el concepto de ataque en lugar del artefacto de herramienta.

2. **Viabilidad estadística**: Etiquetas como "Heartbleed" (11 muestras en el dataset completo) o "Web Attack – Sql Injection" (21 muestras) no pueden formar conjuntos de entrenamiento, validación y test representativos de forma independiente. Su absorción en clases padre como DoS y WebAttack les aporta la masa crítica estadística necesaria.

3. **Relevancia operacional**: Desde el punto de vista de un analista de seguridad, la categoría de ataque (DoS, escaneo, acceso no autorizado por fuerza bruta) determina la respuesta operativa, no la herramienta específica. Las 7 clases canónicas se alinean con esta perspectiva.

#### 3.3.2 Distribución de clases en el dataset

La distribución de clases en el dataset original (2.830.743 filas, antes de limpieza) presenta un desbalanceo severo:

| Clase canónica | Muestras aprox. | Porcentaje |
|---|---|---|
| BENIGN | ~1.648.000 | 58,2 % |
| DoS | ~253.000 | 8,9 % |
| Scan | ~159.000 | 5,6 % |
| DDoS | ~128.000 | 4,5 % |
| BruteForce | ~13.800 | 0,49 % |
| WebAttack | ~2.180 | 0,08 % |
| Botnet | ~1.966 | 0,07 % |

La proporción entre la clase más frecuente (BENIGN) y las menos frecuentes (Botnet, WebAttack) supera **800:1**. Este nivel de desbalanceo excede con creces los escenarios estándar en los que se aplican técnicas básicas de compensación, y exige una combinación de estrategias: pesos de clase en la función de pérdida, muestreo balanceado por batch, y métricas de evaluación insensibles a la clase mayoritaria.

### 3.4 Pipeline de preprocesado

El pipeline de carga y preprocesado, implementado en `utils/loader.py`, transforma el dataset bruto en los tres subconjuntos listos para el entrenamiento mediante siete pasos aplicados en orden estricto:

1. **Carga con encoding `latin-1`** y corrección del carácter en-dash en etiquetas (necesario para que `"Web Attack – XSS"` se mapee correctamente a través de `ATTACK_GROUPS`).
2. **Sustitución de valores Inf y eliminación de NaN** generados por CICFlowMeter al dividir por cero en features de tasa cuando la duración del flujo es cero.
3. **Eliminación de duplicados exactos** para prevenir *data leakage* entre los conjuntos de entrenamiento y test.
4. **Filtrado de clases muy minoritarias** que no superen los umbrales mínimos de muestra, protegiendo contra particiones estadísticamente inviables.
5. **Conversión a float32** para reducir el coste de memoria a la mitad sin pérdida de precisión relevante.
6. **Muestreo estratificado** que preserva la distribución de clases del dataset completo.
7. **Partición doble estratificada** en conjuntos de entrenamiento (70 %), validación (15 %) y test (15 %).

El detalle cuantitativo y la motivación de cada paso se desarrollan en §4.

### 3.5 Análisis exploratorio de datos (EDA)

El EDA generado automáticamente por `utils/loader.py` produce tres figuras guardadas en `output/eda/`:

**Distribución de clases** (`class_distribution.png`): el diagrama de barras confirma visualmente el desbalanceo extremo descrito en §3.3.2. Sobre la muestra estratificada de 1.000.001 filas, BENIGN acumula 831.109 muestras (83,1 %), mientras que Botnet y WebAttack se sitúan con 773 y 850 muestras respectivamente, siendo invisibles a escala en el gráfico de barras. El gráfico de sectores ilustra cómo las cuatro clases de ataque con mayor representación (DoS 7,7 %, DDoS 5,1 %, Scan 3,6 %) apenas ocupan un 16 % del espacio total, con las tres clases minoritarias comprimidas en el segmento restante.

**Correlación entre features** (`feature_correlation.png`): el mapa de calor de Pearson sobre las 20 features de mayor varianza revela dos bloques de alta multicolinealidad. El primero agrupa las features de tiempo de flujo y de IAT (*Inter-Arrival Time*): `Flow Duration`, `Fwd IAT Total`, `Bwd IAT Total`, `Fwd IAT Max`, `Flow IAT Max` y las features de actividad (`Idle Max`, `Idle Mean`, `Idle Min`) presentan correlaciones de Pearson superiores a 0,75 entre sí. El segundo bloque reúne las IAT medias y desviaciones estándar (`Fwd IAT Std`, `Fwd IAT Mean`, `Bwd IAT Mean`, `Flow IAT Std`, `Bwd IAT Std`), también con alta correlación interna. Destaca la práctica identidad de `Fwd Header Length` y `Fwd Header Length.1`, que son la misma feature duplicada por un artefacto de CICFlowMeter. Estos grupos de features redundantes justifican que XGBoost use `colsample_bytree < 1.0` para reducir la varianza, y que la NN use Dropout y BatchNorm como mecanismo de regularización implícita.

**Distribuciones por clase** (`feature_distributions.png`): las curvas KDE por clase para las features con mayor estadístico F de ANOVA muestran separaciones claras entre categorías. Features de actividad como `Idle Mean` separan efectivamente el tráfico BENIGN (flujos con tiempos de inactividad largos) del tráfico DoS/DDoS (flujos de alta intensidad sin pausas). Las features de subflow (`Subflow Bwd Bytes`, `Subflow Fwd Packets`) presentan distribuciones muy distintas entre BruteForce (payloads pequeños y repetitivos) y DDoS (alto volumen en dirección backward), lo que es consistente con que estas features lideren la importancia en XGBoost (§8.6).

---

## 4. Preprocesado y limpieza de datos

El pipeline de carga y preprocesado, implementado en `utils/loader.py`, transforma el dataset bruto en los tres subconjuntos listos para el entrenamiento de los modelos. Esta sección detalla el flujo completo de transformaciones con sus datos cuantitativos, complementando la descripción motivacional de §3.4.

### 4.1 Carga de archivos y corrección de codificación

Los 8 archivos CSV de CICIDS2017 se cargan secuencialmente mediante `pandas.read_csv` con los parámetros `encoding='latin-1'` y `low_memory=False`. El parámetro de codificación es imprescindible: los archivos fueron generados en un entorno con codificación `latin-1`, por lo que usar el `utf-8` por defecto de pandas provoca que el carácter en-dash "–" (U+2013), presente en etiquetas como `"Web Attack – XSS"` o `"Web Attack – Sql Injection"`, se interprete erróneamente como la secuencia *mojibake* `"ï¿½"` (bytes U+00EF, U+00BF, U+00BD). Puesto que el diccionario `ATTACK_GROUPS` de `config.py` usa el en-dash correcto, las filas afectadas quedarían sin mapear y se descartarían silenciosamente. Para evitarlo, se aplica una sustitución explícita `str.replace('ï¿½', '–', regex=False)` sobre la columna `Label` antes de cualquier operación de mapeo.

Tras la carga y concatenación de los 8 archivos, el dataset combinado cuenta con **2.830.743 filas** y **79 columnas** (78 features numéricas + columna `Label`). Los nombres de columna se normalizan eliminando espacios iniciales y finales, ya que CICFlowMeter genera cabeceras con espacios que difieren según el archivo de origen.

### 4.2 Eliminación de valores inválidos y duplicados

La limpieza se aplica en dos pasos secuenciales sobre el dataset combinado:

**Eliminación de NaN e Inf.** CICFlowMeter calcula features de tasa como `Flow Bytes/s = total_bytes / Flow Duration`. Cuando la duración del flujo es cero (conexiones instantáneas o paquetes SYN sin respuesta), esta operación produce `+Inf` o `-Inf`. Estos valores se reemplazan primero por `NaN` mediante `df.replace([np.inf, -np.inf], np.nan)`, y a continuación se eliminan las filas con algún `NaN` en columnas de features. Se eliminan así aproximadamente **2.867 filas** (≈0,1% del total), muy por debajo del umbral del 5% considerado preocupante. La decisión de eliminar en lugar de imputar está motivada por la naturaleza artefactual de los valores Inf: un flujo de duración cero no tiene tasa definida, e imputar con la media introduciría valores ficticios en features de alta discriminancia como `Flow Bytes/s`.

**Eliminación de duplicados.** El dataset contiene un porcentaje elevado de filas exactamente idénticas en las 79 columnas. La eliminación mediante `pandas.DataFrame.drop_duplicates()` —que mantiene la primera ocurrencia y descarta las restantes— elimina aproximadamente **307.078 filas** (≈10,9% del dataset tras la limpieza de NaN). Este paso debe ejecutarse **antes** de la partición train/test para evitar *data leakage*: sin deduplicación previa, la misma fila podría aparecer simultáneamente en los conjuntos de entrenamiento y de test, inflando artificialmente las métricas de evaluación.

El balance total de la etapa de limpieza es:

| Paso | Filas eliminadas | % del original |
|---|---|---|
| NaN/Inf | ~2.867 | ~0,10 % |
| Duplicados | ~307.078 | ~10,85 % |
| **Total eliminado** | **~309.945** | **~10,95 %** |
| **Dataset limpio resultante** | **~2.520.798** | **~89,05 %** |

### 4.3 Filtrado de clases minoritarias

Sobre el dataset limpio, se eliminan las clases que no superen **al menos uno** de los dos umbrales siguientes (condición OR):

- **Umbral absoluto**: menos de `MIN_CLASS_SAMPLES = 100` muestras.
- **Umbral relativo**: menos de `MIN_CLASS_RATIO = 0.0005` (0,05 %) del total.

La lógica OR es deliberadamente conservadora: una clase que falle cualquiera de los dos criterios no tiene masa suficiente para producir particiones train/val/test estadísticamente fiables. Para el dataset CICIDS2017 con las 7 clases canónicas, todas superan ambos umbrales tras la limpieza: incluso las más pequeñas (Botnet y WebAttack) cuentan con más de 1.500 muestras. El filtrado cobra relevancia en configuraciones con umbrales distintos o al trabajar con subconjuntos del dataset, protegiendo contra la transmisión de clases degeneradas al proceso de muestreo y partición.

Este paso se aplica siempre **antes** del muestreo estratificado, garantizando que las clases que llegan al split dispongan de suficientes muestras en los subconjuntos de validación y test.

### 4.4 Conversión a float32

Todas las features numéricas se convierten de `float64` (tipo por defecto de pandas) a `float32`. Con 78 features y 1.000.001 filas en la muestra de trabajo, la representación `float64` requiere aproximadamente **593 MB de RAM** solo para la matriz de features; el cast a `float32` reduce este coste a **297 MB**, sin pérdida de precisión relevante para los modelos del proyecto. La precisión de 7 dígitos decimales significativos de `float32` es más que suficiente para features como conteos de paquetes o tasas de bytes, donde la variabilidad numérica está en órdenes de magnitud muy superiores a esa precisión.

### 4.5 Muestreo estratificado

Del dataset limpio (~2,52 millones de filas) se extrae una **muestra estratificada de 1.000.001 filas** mediante `groupby().apply(lambda: sample(frac=...))`. El muestreo estratificado preserva la distribución de clases original: cada clase contribuye con la misma proporción que tiene en el dataset completo, garantizando que las clases minoritarias (Botnet, WebAttack) no queden infrarrepresentadas. Un mecanismo de compensación de redondeo añade filas del conjunto no muestreado si la suma por clase resulta inferior al tamaño objetivo.

La elección de 1.000.001 filas obedece a un compromiso entre representatividad estadística y coste computacional. Con esta muestra las clases más pequeñas (Botnet: 773, WebAttack: 850) disponen de suficiente masa para producir particiones train/val/test con más de 100 ejemplos por subconjunto en las clases minoritarias, mejorando la fiabilidad de las métricas y reduciendo la varianza de las fronteras de decisión aprendidas. El coste en tiempo de entrenamiento es asumible dado el hardware disponible: la red neuronal NumPy entrenó en ~39 minutos y XGBoost con tuning completo en ~85 minutos sobre CPU.

### 4.6 Partición estratificada doble (70/15/15)

La muestra de 1.000.001 filas se divide en tres subconjuntos mediante **dos splits secuenciales estratificados** con `sklearn.model_selection.train_test_split`:

1. **Split 1**: separación del test set (15 %) del resto (85 %), con `stratify=Label`.
2. **Split 2**: el 85 % restante se divide en train (82,35 % de ese 85 %, equivalente al 70 % del total) y validación (17,65 % del 85 %, equivalente al 15 % del total), con `stratify=Label`.

Los tamaños resultantes son: **train = 700.000 filas** | **validación = 150.000 filas** | **test = 150.001 filas**.

### 4.7 Normalización de features

El pipeline externo de `utils/loader.py` **no aplica ninguna normalización** de features. Esto es correcto para los modelos basados en árboles: tanto el Isolation Forest como XGBoost son invariantes a las escalas de las features por su naturaleza de partición por umbrales, por lo que StandardScaler no aportaría ningún beneficio.

La red neuronal, en cambio, es sensible a la escala de las entradas. La clase `NeuralNetwork` de `models/neural_network.py` **incorpora un `StandardScaler` interno** (`self._scaler: StandardScaler = StandardScaler()`). Durante el entrenamiento (`fit()`), el escalador se ajusta exclusivamente sobre las **muestras anómalas del conjunto de entrenamiento** —las enrutadas desde Stage 1— mediante `fit_transform`, y se aplica a las muestras de validación y test mediante `transform`. Este diseño encapsula la normalización dentro del modelo Stage 2 y previene el *data leakage*: el escalador nunca ve muestras de validación ni de test en su fase de ajuste.

---

## 5. Técnicas y modelos aplicados

### 5.1 Arquitectura en cascada: motivación y flujo de decisión

La arquitectura en cascada divide el problema de detección de intrusiones en dos subproblemas conceptualmente distintos. El flujo de decisión en inferencia es el siguiente:

```
X_entrada → IF.predict_binary()
              ├─ score ≥ umbral  →  BENIGN  (etiqueta directa, sin Stage 2)
              └─ score < umbral  →  Anómalo → Clasificador multi-clase
                                                   → Etiqueta de ataque (o BENIGN)
```

En el entrenamiento, el Isolation Forest se ajusta **exclusivamente sobre las muestras BENIGN** del conjunto de entrenamiento, aprendiendo el perfil estadístico del tráfico legítimo sin haber visto ningún ejemplo de ataque. Esta distinción es fundamental: si el IF se entrenara sobre todas las clases, aprendería a separar BENIGN del resto de forma implícitamente supervisada, comprometiendo su capacidad de generalizar a tipos de ataque no vistos durante el entrenamiento.

En inferencia, todos los flujos de red se evalúan primero por el IF. Los que superan el umbral se etiquetan directamente como BENIGN. Los que caen por debajo se consideran anómalos y se enrutan al clasificador de Stage 2 (XGBoost o red neuronal), que determina el tipo específico de ataque entre las 7 clases canónicas. El espacio de salida de ambos clasificadores de Stage 2 **incluye la clase BENIGN**, lo que les permite **corregir falsos positivos del IF**: flujos legítimos que el IF clasificó erróneamente como anómalos pueden ser devueltos a BENIGN por el Stage 2.

Un aspecto crítico son los **falsos negativos del Stage 1**: ataques que el IF clasifica como BENIGN (score ≥ umbral). Estos flujos nunca llegan al Stage 2 y se etiquetan como BENIGN en la salida final del pipeline, propagando el error de forma irrecuperable. La optimización del umbral con `beta=2.0` busca minimizar estos falsos negativos a costa de generar más falsas alarmas (que Stage 2 puede corregir).

### 5.2 Stage 1 — Isolation Forest

El *Isolation Forest* (Liu et al., 2008) es un algoritmo de detección de anomalías que explota el principio de que las observaciones atípicas son más fáciles de aislar que las normales. El algoritmo construye un *ensemble* de **árboles de aislamiento** (*isolation trees*): en cada árbol se realizan particiones binarias aleatorias del espacio de features —seleccionando aleatoriamente una feature y un punto de corte dentro de su rango— hasta que la muestra queda aislada en un nodo hoja o se alcanza la profundidad máxima.

La **puntuación de anomalía** se deriva de la **profundidad promedio de aislamiento** en el bosque: muestras que requieren pocas particiones para ser aisladas ocupan regiones escasas del espacio de features y se consideran anómalas. En scikit-learn, la función `decision_function` devuelve el negativo de esta puntuación, de modo que valores más negativos corresponden a muestras más anómalas.

Los hiperparámetros del IF en este proyecto son:

| Parámetro | Valor | Justificación |
|---|---|---|
| `n_estimators` | 200 | 200 árboles; suficiente para estabilizar las estimaciones de profundidad |
| `max_samples` | `"auto"` | min(256, n_samples) por árbol; estándar según el artículo original |
| `contamination` | 0.15 | Estimación del porcentaje de anomalías para el offset interno de sklearn |
| `max_features` | 1.0 | Usa todas las features en cada partición aleatoria |
| `bootstrap` | False | Submuestreo sin reemplazamiento |
| `n_jobs` | -1 | Paralelismo completo usando todos los núcleos disponibles |

El parámetro `contamination=0.15` merece una aclaración. La fracción real de tráfico no-BENIGN en CICIDS2017 es aproximadamente el 41,8%; sin embargo, `contamination` en scikit-learn ajusta el offset del método `predict()` interno de sklearn, que en este proyecto **no se utiliza**: las predicciones se basan en un umbral personalizado optimizado sobre el conjunto de validación. El valor 0.15 actúa como estimación de partida sin impacto directo en las predicciones finales del pipeline.

**Optimización del umbral.** Tras el entrenamiento, se ejecuta un barrido de **300 candidatos de umbral** uniformemente distribuidos en el rango `[min_score, max_score]` de las puntuaciones del conjunto de validación (`np.linspace(scores.min(), scores.max(), num=300)`). Para cada umbral candidato $t$, la regla de decisión es: `score(x) < t` → anómalo; `score(x) ≥ t` → BENIGN. Se calcula el F-beta con $\beta = 2.0$ para cada candidato:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}$$

y se selecciona el umbral que maximiza esta métrica. La elección de $\beta = 2.0$ pondera el recall el doble que la precisión, lo que es correcto para un IDS: no detectar un ataque (falso negativo) tiene consecuencias operativas mucho más graves que generar una falsa alarma (falso positivo), que el Stage 2 puede filtrar.

El modelo entrenado se persiste como `isolation_forest.joblib` (objeto sklearn serializado con joblib) y `if_threshold.npy` (umbral óptimo como array NumPy de un elemento).

### 5.3 Stage 2A — XGBoost

XGBoost (*Extreme Gradient Boosting*, Chen & Guestrin, 2016) es un método de *ensemble* de árboles de decisión entrenados mediante **potenciación por gradiente** (*gradient boosting*): en cada iteración, se añade un nuevo árbol que modela el gradiente negativo de la función de pérdida respecto a las predicciones acumuladas, corrigiendo el residuo de la iteración anterior. El resultado es una suma ponderada de árboles débiles que minimiza conjuntamente la pérdida de entrenamiento y un término de regularización.

**Hiperparámetros por defecto** (activos con `--skip-tuning`):

| Parámetro | Valor | Descripción |
|---|---|---|
| `n_estimators` | 300 | Número máximo de árboles |
| `max_depth` | 6 | Profundidad máxima de cada árbol |
| `learning_rate` | 0.1 | Tasa de contracción (*shrinkage*) por árbol |
| `subsample` | 0.9 | Fracción del dataset de entrenamiento muestreada por árbol |
| `colsample_bytree` | 0.9 | Fracción de features muestreada por árbol |
| `reg_alpha` | 0.0 | Regularización L1 sobre pesos de hoja |
| `reg_lambda` | 1.0 | Regularización L2 sobre pesos de hoja |
| `eval_metric` | `"mlogloss"` | Métrica de evaluación para early stopping |
| `early_stopping_rounds` | 20 | Paciencia para early stopping |
| `tree_method` | `"hist"` | Construcción basada en histogramas (compatible CPU y GPU) |

**Búsqueda de hiperparámetros.** Cuando `--skip-tuning` no está activo, se ejecuta un `RandomizedSearchCV` con **20 iteraciones** y **validación cruzada estratificada de 3 folds** (`StratifiedKFold`) sobre las muestras anómalas del conjunto de entrenamiento. La métrica de selección es `f1_macro` sobre las clases de ataque. El espacio de búsqueda comprende `n_estimators` ∈ {200, 300, 500}, `max_depth` ∈ {4, 6, 8}, `learning_rate` ∈ {0.05, 0.1, 0.2}, `subsample` ∈ {0.8, 1.0} y `colsample_bytree` ∈ {0.8, 1.0}. Los mejores hiperparámetros se fusionan con los valores por defecto antes del entrenamiento final (los valores encontrados por la búsqueda tienen precedencia).

**Pesos de clase.** El desbalanceo entre clases de ataque exige compensación activa. Para cada muestra se asigna un **peso inversamente proporcional a la frecuencia de su clase**:

$$w_c = \frac{1}{f_c \cdot K}, \quad \text{normalizado para que } \bar{w} = 1$$

donde $f_c$ es la frecuencia relativa de la clase $c$ y $K$ el número de clases. Este esquema equivale a la estrategia `class_weight='balanced'` de scikit-learn y garantiza que las clases minoritarias (Botnet, WebAttack) tengan el mismo impacto esperado en el gradiente que las mayoritarias.

**Early stopping.** El modelo final se entrena con `early_stopping_rounds=20`: si la pérdida `mlogloss` sobre el conjunto de validación de anomalías no mejora en 20 iteraciones consecutivas, el entrenamiento se detiene y se usa el mejor modelo hasta ese punto. Esto previene el sobreajuste sin necesidad de un número fijo de estimadores.

Los artefactos del modelo se persisten como `xgboost.joblib` (clasificador serializado) y `xgb_label_encoder.joblib` (codificador de etiquetas para restaurar predicciones a strings canónicos).

### 5.4 Stage 2B — Red Neuronal NumPy

El Pipeline B sustituye XGBoost por un perceptrón multicapa (*Multi-Layer Perceptron*, MLP) implementado íntegramente en NumPy, sin el uso de ninguna biblioteca de aprendizaje profundo. Esta elección exige implementar manualmente el ciclo de entrenamiento completo: *forward pass*, *backpropagation* por regla de la cadena, optimizador Adam, Batch Normalization con estadísticas diferenciadas para entrenamiento e inferencia, Dropout invertido, Focal Loss con suavizado de etiquetas, y planificación del *learning rate*.

#### Arquitectura

La red tiene cuatro capas lineales con el siguiente patrón en las tres capas ocultas: transformación lineal → Batch Normalization → ReLU → Dropout invertido.

```
Entrada (78 features)
  → Dense(256) → BatchNorm → ReLU → Dropout(p=0.15)
  → Dense(128) → BatchNorm → ReLU → Dropout(p=0.15)
  → Dense(64)  → BatchNorm → ReLU → Dropout(p=0.10)
  → Dense(7 clases) → Softmax
```

Los pesos de las capas lineales se inicializan con **inicialización He**: $W \sim \mathcal{N}(0, \sqrt{2/n_{\text{in}}})$, adecuada para activaciones ReLU ya que compensa la pérdida de varianza que introduce dicha activación. Los sesgos se inicializan a cero; los parámetros aprendibles de BN, $\gamma$ (escala) y $\beta$ (desplazamiento), se inicializan a 1 y 0 respectivamente.

#### Batch Normalization

La *Batch Normalization* (BN) se aplica en cada capa oculta, después de la transformación lineal y antes de la activación ReLU. En **modo entrenamiento**, normaliza la activación del mini-batch:

$$\hat{z} = \frac{z - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \qquad \text{out} = \gamma \hat{z} + \beta$$

donde $\mu_B$ y $\sigma_B^2$ son la media y varianza del mini-batch, $\varepsilon = 10^{-5}$ es la constante de estabilidad numérica, y $\gamma, \beta$ son los parámetros aprendibles. Simultáneamente, se actualizan **estadísticas running** mediante suavizado exponencial con momento 0.9:

$$\mu_{\text{run}} \leftarrow 0.9 \cdot \mu_{\text{run}} + 0.1 \cdot \mu_B; \qquad \sigma^2_{\text{run}} \leftarrow 0.9 \cdot \sigma^2_{\text{run}} + 0.1 \cdot \sigma^2_B$$

En **modo inferencia**, se usan las estadísticas running acumuladas durante el entrenamiento, garantizando normalización determinista sin depender del tamaño o composición del batch. La BN estabiliza el gradiente y reduce la dependencia del rendimiento respecto a la escala del *learning rate*, lo que es especialmente beneficioso con el desbalanceo severo presente en este dataset.

#### Dropout invertido

El *Dropout* se aplica tras la activación ReLU de cada capa oculta con tasas `[0.15, 0.15, 0.10]`. Se usa la variante **invertida** (*inverted dropout*): durante el entrenamiento, las activaciones que sobreviven se escalan por $1/(1-p)$, de modo que su esperanza se mantiene igual que en inferencia. Esto elimina la necesidad de cualquier corrección de escala en el modo de predicción. La propagación del gradiente a través del Dropout es trivial: las unidades eliminadas reciben gradiente cero.

#### Función de pérdida: Focal Loss con suavizado de etiquetas

La función de pérdida combina dos componentes para hacer frente al desbalanceo extremo:

**Focal Loss** con $\gamma = 2.0$: cada muestra recibe un peso $(1 - p_t)^\gamma$, donde $p_t$ es la probabilidad predicha para la clase verdadera:

$$\text{FL}(x) = -(1 - p_t)^\gamma \cdot \log(p_t)$$

Este factor reduce el peso de los ejemplos fáciles (bien clasificados, $p_t$ alta) y amplifica el de los difíciles (mal clasificados, $p_t$ baja). Con $\gamma = 2$, un ejemplo con $p_t = 0.9$ contribuye con peso relativo $(0.1)^2 = 0.01$, frente a uno con $p_t = 0.5$ que contribuye con $(0.5)^2 = 0.25$. Así, la clase mayoritaria (BENIGN), una vez bien aprendida, deja de dominar el gradiente, permitiendo que la red se centre en las clases minoritarias. Los pesos de clase adicionales (`use_class_weights=True`) actúan como el factor $\alpha$ de la Focal Loss original, multiplicando el peso focal por clase.

**Suavizado de etiquetas** (*label smoothing*) con $\varepsilon = 0.05$: reemplaza la distribución *one-hot* dura por una distribución suavizada:

$$y_k^{\text{smooth}} = (1-\varepsilon) \cdot \mathbf{1}[k = c] + \frac{\varepsilon}{K}$$

donde $c$ es la clase verdadera y $K$ el número de clases. Esto penaliza la sobreconfianza del modelo (probabilidades cercanas a 1) y mejora la calibración de las predicciones de salida.

#### Optimizador Adam con regularización L2

Los parámetros se actualizan con el optimizador Adam (Kingma & Ba, 2015):

| Parámetro | Valor |
|---|---|
| `learning_rate` | $10^{-3}$ |
| `beta1` | 0.9 |
| `beta2` | 0.999 |
| `epsilon` | $10^{-8}$ |
| `weight_decay` | $10^{-4}$ |

La regularización L2 (`weight_decay`) se aplica como término aditivo en el gradiente de las matrices de pesos antes de la actualización: $g_W \leftarrow g_W + \lambda W$. Este esquema de *decoupled weight decay* evita interacciones indeseadas entre la regularización y el escalado adaptativo de Adam.

#### Muestreo balanceado por batch

Con `use_balanced_batching=True`, los mini-batches se construyen garantizando representación uniforme de las clases: se seleccionan `batch_size // n_classes` muestras por clase (con sobremuestreo con reemplazamiento para clases con pocas muestras disponibles). El número de batches por época se limita al máximo natural de la clase mayoritaria para evitar múltiples pasadas sobre el dataset en una sola época.

Adicionalmente, antes del primer epoch se aplica un **sobremuestreo de inicialización**: las clases minoritarias se replican con ruido gaussiano $\mathcal{N}(0, 0.01)$ en el espacio normalizado hasta representar al menos `oversample_min_ratio = 0.15` veces el tamaño de la clase mayoritaria, evitando copias exactas.

#### Planificación del *learning rate*

El esquema combina dos estrategias:

1. **Calentamiento lineal** (*linear warmup*) durante las primeras `lr_warmup_epochs = 5` épocas: el *learning rate* crece linealmente desde `lr_min = 1e-5` hasta `learning_rate = 1e-3`, evitando actualizaciones grandes al inicio cuando los momentos de Adam aún no están calibrados.

2. **Reducción en plateau** (*ReduceLROnPlateau*): si el F1-macro de validación no mejora en `lr_scheduler_patience = 5` épocas consecutivas, el *learning rate* se multiplica por `lr_scheduler_factor = 0.5`, con mínimo `lr_min = 1e-5`.

#### Early stopping con restauración de mejores pesos

El entrenamiento se detiene si el F1-macro de validación no mejora en `early_stopping_patience = 20` épocas consecutivas. En cada época en que se logra una mejora, se guarda una copia de todos los parámetros en RAM. Al activarse el early stopping o al completar las `epochs = 100` épocas máximas, los parámetros se restauran a la mejor copia guardada, garantizando que el modelo final corresponde al mejor punto de validación observado.

#### Retropropagación manual y recorte de gradiente

El gradiente se propaga capa a capa mediante la regla de la cadena implementada explícitamente en NumPy, sin autograd. El par Softmax + entropía cruzada tiene el gradiente analítico combinado $\nabla_{z_4} \mathcal{L} = (\hat{y} - y^{\text{smooth}}) / N$, numéricamente estable. El gradiente de cada capa oculta traversa en orden inverso las operaciones de Dropout, ReLU y BatchNorm. Para la BN se implementa la fórmula exacta de la regla de la cadena que tiene en cuenta la dependencia de la media y varianza del batch respecto a todos los inputs del mini-batch.

Antes de la actualización de Adam se aplica **recorte de gradiente** (*gradient clipping*) con norma máxima `grad_clip_norm = 1.0`: si la norma L2 del gradiente concatenado de todos los parámetros supera 1.0, los gradientes se escalan uniformemente hasta que la norma resultante sea exactamente 1.0, previniendo explosiones de gradiente en épocas tempranas o ante batches muy desequilibrados.

Los artefactos del modelo se persisten como `nn_weights.npz` (pesos y parámetros BN aprendibles), `nn_bn_running.npz` (estadísticas running de BN para inferencia), `nn_label_encoder.joblib` y `nn_scaler.joblib`.

---

## 6. Metodología experimental

### 6.1 Reproducibilidad y semilla global

Toda la experimentación se controla mediante la semilla global `seed = 42`, aplicada en tres capas del stack:

- **Red neuronal**: la clase `NeuralNetwork` usa `np.random.default_rng(42)` para todas las operaciones estocásticas del modelo: inicialización de pesos He, Dropout, muestreo balanceado por batch y sobremuestreo de clases minoritarias.
- **scikit-learn**: se pasa `random_state=42` a `train_test_split`, `StratifiedKFold`, `IsolationForest` y `RandomizedSearchCV`, garantizando particiones y búsquedas idénticas entre ejecuciones.
- **XGBoost**: se pasa `random_state=42` al constructor de `XGBClassifier`.

Con estas semillas fijadas, la ejecución completa del experimento —desde la partición del dataset hasta las métricas finales sobre el test set— es **reproducible bit a bit**.

### 6.2 Flujo completo del experimento

El punto de entrada es `main.py`, que orquesta las siguientes etapas en orden:

1. **Carga y preprocesado** (`utils/loader.py`): carga los 8 CSVs, limpieza, filtrado de clases, cast a float32.
2. **EDA** (`utils/loader.py`): generación de figuras exploratorias guardadas en `output/eda/`.
3. **Partición** (`utils/loader.py`): muestreo estratificado a 1.000.001 filas y split 70/15/15 (700.000 / 150.000 / 150.001).
4. **Pipeline A** (`pipelines/pipeline_a.py`): entrenamiento de IF + XGBoost, evaluación sobre test, guardado en `output/pipeline_a/`.
5. **Pipeline B** (`pipelines/pipeline_b.py`): entrenamiento de IF + NN, evaluación sobre test, guardado en `output/pipeline_b/`.
6. **Comparativa**: cálculo y registro de métricas comparativas entre Pipeline A y Pipeline B sobre el mismo test set, en `output/results.txt`.

### 6.3 Flags de control de ejecución

**`--skip-tuning`**: suprime la búsqueda `RandomizedSearchCV` en Pipeline A, entrenando XGBoost directamente con los hiperparámetros por defecto de `XGBoostConfig` en `config.py`. Para Pipeline B no tiene efecto, ya que la red neuronal no dispone de búsqueda automatizada de hiperparámetros: la arquitectura y los valores de `NeuralNetworkConfig` son fijos. Este flag es útil para reproducir resultados rápidamente sin incurrir en el coste de 20 iteraciones × 3 folds de validación cruzada.

**`--load-model`**: cuando se activa, Pipeline B no se reentrena; en su lugar, se carga el modelo previamente guardado desde `output/pipeline_b/`, restaurando el Isolation Forest (`isolation_forest.joblib` + `if_threshold.npy`) y la red neuronal (`nn_weights.npz`, `nn_bn_running.npz`, `nn_label_encoder.joblib`, `nn_scaler.joblib`) sin ejecutar ningún ciclo de entrenamiento. Esto permite evaluar o comparar un modelo ya entrenado sin incurrir en el coste computacional de la NN.

### 6.4 Estrategia de validación del Isolation Forest

El barrido de umbral del IF se realiza sobre el conjunto de validación, que debe contener tanto muestras BENIGN (clase negativa) como de ataque (clase positiva) para que la curva F-beta sea informativa. En los casos excepcionales en que el IF no genera ninguna muestra anómala en el conjunto de validación —posible con configuraciones de `contamination` extremas o datasets muy reducidos—, se activa un **mecanismo de reserva** (*fallback*): se toma el 10 % de las muestras de entrenamiento anómalas como conjunto de pseudo-validación, satisfaciendo los requisitos tanto del optimizador de umbral como del early stopping del Stage 2.

### 6.5 Paralelismo, hardware y guardado de modelos

**Paralelismo.** El Isolation Forest usa `n_jobs=-1`, aprovechando todos los núcleos disponibles para construir los 200 árboles en paralelo. XGBoost usa igualmente `n_jobs=-1` con `tree_method='hist'`, que habilita construcción de árboles por histogramas optimizada para CPU. La red neuronal opera sobre matrices NumPy con operaciones BLAS, que aprovechan paralelismo SIMD a nivel de instrucción sin paralelismo explícito en el código de alto nivel.

**Hardware y entorno.** El experimento está diseñado para ejecutarse en **CPU únicamente**: la red neuronal no usa CUDA ni ninguna aceleración por GPU. El entorno de referencia es Python 3.10. Las dependencias principales son NumPy, pandas, scikit-learn, XGBoost, matplotlib y seaborn; no se utiliza ninguna biblioteca de aprendizaje profundo.

**Guardado de modelos.** Los artefactos de cada pipeline se almacenan en subdirectorios de `output/`:

| Modelo | Archivo(s) | Formato |
|---|---|---|
| Isolation Forest (modelo) | `isolation_forest.joblib` | joblib |
| Isolation Forest (umbral) | `if_threshold.npy` | NumPy array |
| XGBoost (clasificador) | `xgboost.joblib` | joblib |
| XGBoost (codificador) | `xgb_label_encoder.joblib` | joblib |
| Red Neuronal (pesos + params BN) | `nn_weights.npz` | NumPy npz |
| Red Neuronal (estadísticas BN) | `nn_bn_running.npz` | NumPy npz |
| Red Neuronal (encoder + scaler) | `nn_label_encoder.joblib`, `nn_scaler.joblib` | joblib |

---

## 7. Métricas utilizadas para la evaluación

### 7.1 F1-macro como métrica primaria

La elección de la métrica de evaluación principal es crítica cuando las clases presentan el desbalanceo pronunciado de CICIDS2017. La *accuracy* global —fracción de predicciones correctas sobre el total de muestras— no es una métrica adecuada para este problema: dado que BENIGN representa el 83,1 % del conjunto de test, un clasificador trivial que etiquetara toda muestra como BENIGN obtendría una accuracy del 83,1 % sin detectar ningún ataque.

La **F1-macro** se adopta como métrica primaria de comparación entre pipelines. Se calcula como la media aritmética no ponderada de los F1 de cada clase:

$$\text{F1-macro} = \frac{1}{K} \sum_{c=1}^{K} F1_c = \frac{1}{K} \sum_{c=1}^{K} \frac{2 \cdot \text{precision}_c \cdot \text{recall}_c}{\text{precision}_c + \text{recall}_c}$$

donde $K = 7$ es el número de clases. La característica clave de esta métrica es que **cada clase contribuye por igual al resultado final**, independientemente de su tamaño. Esto significa que un F1 bajo en Botnet (116 muestras en test) penaliza en igual medida que un F1 bajo en BENIGN (124.667 muestras), lo que obliga a los modelos a desarrollar capacidad discriminativa en todas las categorías de ataque, incluidas las más escasas.

La **F1-weighted** (media ponderada por el soporte de cada clase) y la **accuracy global** se reportan también como métricas complementarias, útiles para contextualizar el rendimiento en el dominio mayoritario (tráfico BENIGN), pero no se emplean como criterio de comparación principal entre pipelines.

### 7.2 Métricas del Stage 1: el Isolation Forest como filtro binario

El Isolation Forest opera como un clasificador binario: BENIGN frente a Anómalo. Para evaluarlo se emplean cuatro métricas complementarias sobre la partición binaria del test set:

**Recall** (sensibilidad, tasa de verdaderos positivos): fracción de ataques reales correctamente identificados como anómalos:

$$\text{Recall} = \frac{TP}{TP + FN}$$

En un IDS, el recall es la métrica más crítica: un falso negativo (ataque clasificado como BENIGN) implica que ningún componente del sistema tendrá oportunidad de detectarlo, ya que los flujos etiquetados como BENIGN por Stage 1 no alcanzan Stage 2. Por este motivo, el umbral de Stage 1 se optimiza con F-beta ($\beta=2$), que pondera el recall el doble que la precisión.

**Tasa de falsos negativos** (FNR, *False Negative Rate*): complemento del recall ($\text{FNR} = 1 - \text{Recall}$); representa la proporción de ataques que el sistema pierde de forma irrecuperable.

**Tasa de falsos positivos** (FPR, *False Positive Rate*): fracción de flujos benignos incorrectamente marcados como anómalos y enviados al Stage 2:

$$\text{FPR} = \frac{FP}{FP + TN}$$

Un FPR alto no implica directamente predicciones incorrectas en el sistema completo —el Stage 2 puede corregir los falsos positivos del Stage 1— pero sí incrementa la carga computacional del Stage 2 y aumenta el riesgo de error en la segunda etapa.

**Precisión del Stage 1**: fracción de flujos marcados como anómalos que son efectivamente ataques. Informa sobre la selectividad del detector de anomalías.

### 7.3 AUC-ROC por clase (One-vs-Rest)

El **Área Bajo la Curva ROC** (*AUC-ROC*) por clase, calculado mediante la estrategia *One-vs-Rest* (OvR), mide la capacidad discriminativa del clasificador de Stage 2 a todos los posibles umbrales de decisión. Para la clase $c$, el clasificador se evalúa como binario (clase $c$ vs. resto), y la curva ROC representa el trade-off entre la TPR y la FPR a lo largo de todos los umbrales posibles. El AUC-OvR es insensible al desbalanceo de clases, ya que el área refleja la probabilidad de que el modelo asigne mayor puntuación a una muestra positiva aleatoria que a una negativa aleatoria. Las curvas ROC por clase para ambos pipelines se visualizan en `output/comparison/roc_curves.png`.

### 7.4 Curvas Precision-Recall

Las curvas Precision-Recall (PR) ofrecen una perspectiva complementaria a las curvas ROC en escenarios con clases muy desbalanceadas: cuando la clase negativa es muy mayoritaria, la FPR puede resultar baja incluso con muchos falsos positivos en valor absoluto, haciendo que las curvas ROC sean optimistas. Las curvas PR, al relacionar directamente precision y recall sin involucrar los verdaderos negativos, son más informativas en estos escenarios. Están disponibles en `output/comparison/pr_curves.png`.

### 7.5 Matriz de confusión como herramienta diagnóstica

La **matriz de confusión** (dimensión $7 \times 7$) detalla el número de muestras de cada clase real que el sistema predice en cada clase predicha. A diferencia de las métricas agregadas, permite identificar patrones de error específicos: qué tipos de ataque se confunden entre sí, qué clases se sobre-predicen (*over-prediction*) y qué clases se infradetectan. Esta información es crítica para entender el impacto del desbalanceo en las fronteras de decisión aprendidas. Las matrices de confusión de ambos pipelines se encuentran en `output/pipeline_a/confusion_matrix.png` y `output/pipeline_b/confusion_matrix.png`.

---

## 8. Resultados obtenidos

### 8.1 Configuración final del experimento

El experimento se ejecutó el 7 de junio de 2026 con la configuración completa (`--pipeline all`, con EDA y tuning activos). Los parámetros de ejecución son:

| Parámetro | Valor |
|---|---|
| Semilla global | 42 |
| Muestra estratificada | 1.000.001 filas |
| Split train / val / test | 700.000 / 150.000 / 150.001 |
| Beta (IF, optimización de umbral) | 2,0 |
| Dataset limpio (tras NaN/Inf + deduplicación) | 2.520.798 filas |
| Tiempo total del experimento | 7.637 s (~2 h 7 min) |

La distribución del conjunto de test (150.001 filas) por clase es:

| Clase | Muestras test | Porcentaje |
|---|---|---|
| BENIGN | 124.667 | 83,11 % |
| DoS | 11.530 | 7,69 % |
| DDoS | 7.617 | 5,08 % |
| Scan | 5.399 | 3,60 % |
| BruteForce | 545 | 0,36 % |
| WebAttack | 127 | 0,08 % |
| Botnet | 116 | 0,08 % |

### 8.2 Resultados del Stage 1 — Isolation Forest

Ambos pipelines comparten exactamente el mismo Isolation Forest: se entrena una sola vez sobre las muestras BENIGN del conjunto de entrenamiento y se optimiza sobre el conjunto de validación. Las métricas de Stage 1 son, por tanto, idénticas para Pipeline A y Pipeline B:

| Métrica | Valor |
|---|---|
| Recall (TPR) | 0,9682 |
| FNR | 0,0318 |
| Precisión | 0,2467 |
| FPR | 0,6008 |

El umbral óptimo seleccionado por el barrido de F-beta sobre el conjunto de validación se encuentra persistido en `output/pipeline_a/model/if_threshold.npy`. La curva de umbral completa (recall vs. FPR a lo largo de los 300 candidatos) está disponible en `output/pipeline_a/threshold_curve.png`.

El recall del 96,82 % indica que el IF detecta como anómalos casi el 97 % de todos los flujos de ataque del test. El FNR del 3,18 % representa los ataques que el IF clasifica erróneamente como BENIGN y que, por tanto, nunca alcanzan Stage 2. Este nivel de recall es coherente con el objetivo del umbral optimizado con $\beta=2,0$.

El FPR del 60,08 % es la contrapartida directa de la alta sensibilidad: el IF etiqueta como anómalos el 60 % de los flujos realmente benignos. Esto implica que Stage 2 recibe aproximadamente 74.800 flujos benignos adicionales que debe reclasificar correctamente como BENIGN. La curva de umbral con el trade-off entre recall y FPR está disponible en `output/pipeline_a/threshold_curve.png`.

### 8.3 Resultados del Stage 2 — Métricas globales

| Métrica | Pipeline A (IF + XGBoost) | Pipeline B (IF + NN NumPy) |
|---|---|---|
| Accuracy | **0,9936** | 0,9685 |
| F1-macro | **0,9516** | 0,7497 |
| F1-weighted | **0,9936** | 0,9736 |
| Tiempo de ejecución | ~5.102 s (~85 min) | ~2.331 s (~39 min) |

Pipeline A supera el benchmark de F1-macro > 0,90 establecido en los objetivos del proyecto. Pipeline B, con un F1-macro de 0,7497, no alcanza el objetivo de referencia de 0,80; las causas de este gap son identificables y se analizan en §9.

### 8.4 Resultados por clase

| Clase | F1 Pipeline A | F1 Pipeline B | Soporte test |
|---|---|---|---|
| BENIGN | 0,9962 | 0,9808 | 124.667 |
| BruteForce | 0,9909 | 0,9595 | 545 |
| DDoS | 0,9836 | 0,9818 | 7.617 |
| DoS | 0,9762 | 0,9627 | 11.530 |
| Scan | 0,9914 | 0,8521 | 5.399 |
| WebAttack | 0,9615 | 0,3547 | 127 |
| Botnet | 0,7616 | 0,1558 | 116 |

Los informes de clasificación completos con precisión y recall por clase son los siguientes.

**Pipeline A (IF + XGBoost):**

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BENIGN | 0,99 | 1,00 | 1,00 | 124.667 |
| Botnet | 0,62 | 0,99 | 0,76 | 116 |
| BruteForce | 0,99 | 0,99 | 0,99 | 545 |
| DDoS | 1,00 | 0,97 | 0,98 | 7.617 |
| DoS | 1,00 | 0,96 | 0,98 | 11.530 |
| Scan | 0,99 | 0,99 | 0,99 | 5.399 |
| WebAttack | 0,94 | 0,98 | 0,96 | 127 |
| — | — | — | — | — |
| **Macro avg** | **0,93** | **0,98** | **0,95** | — |
| **Weighted avg** | **0,99** | **0,99** | **0,99** | — |

**Pipeline B (IF + NN NumPy):**

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BENIGN | 0,99 | 0,97 | 0,98 | 124.667 |
| Botnet | 0,08 | 0,99 | 0,16 | 116 |
| BruteForce | 0,94 | 0,98 | 0,96 | 545 |
| DDoS | 1,00 | 0,97 | 0,98 | 7.617 |
| DoS | 0,97 | 0,95 | 0,96 | 11.530 |
| Scan | 0,74 | 1,00 | 0,85 | 5.399 |
| WebAttack | 0,22 | 1,00 | 0,35 | 127 |
| — | — | — | — | — |
| **Macro avg** | **0,71** | **0,98** | **0,75** | — |
| **Weighted avg** | **0,98** | **0,97** | **0,97** | — |

La comparación visual del F1 por clase entre ambos pipelines está disponible en `output/comparison/f1_per_class.png` y `output/comparison/metrics_bar.png`.

### 8.5 Curvas de entrenamiento de la red neuronal

Las curvas de entrenamiento (`output/pipeline_b/training_curves.png`) muestran la evolución de la *Cross-Entropy Loss* y la *accuracy* por época para los conjuntos de entrenamiento y validación. Se observan los siguientes patrones relevantes:

**Convergencia rápida y estabilización.** Tanto el *train loss* como el *val loss* descienden bruscamente en las primeras 5-10 épocas y se estabilizan a partir de la época 20, sin variaciones apreciables hasta el final del entrenamiento. Esto indica que el calentamiento lineal del *learning rate* (épocas 1-5) previno actualizaciones desestabilizadoras en la fase inicial.

**Inversión del loss entre conjuntos.** El *val loss* (~0,37 al estabilizarse) se mantiene sistemáticamente **por debajo** del *train loss* (~0,65). Este comportamiento aparentemente anómalo se explica por el uso de *Focal Loss* con muestreo balanceado durante el entrenamiento: los batches de train están sobrerepresentados en ejemplos difíciles (clases minoritarias), lo que eleva la pérdida media de entrenamiento artificialmente. La validación, en cambio, evalúa sobre la distribución real del dataset (83 % BENIGN), donde los ejemplos fáciles dominan y la pérdida es menor.

**Accuracy de validación inferior a la de entrenamiento.** El *train accuracy* alcanza el ~99 % desde la época 5 y se mantiene estable, mientras que el *val accuracy* evoluciona más lentamente, llegando al ~96-97 % hacia la época 40 y estabilizándose sin mejorar significativamente. La brecha de ~2-3 puntos porcentuales entre train y val indica un leve sobreajuste a los patrones del conjunto de entrenamiento balanceado.

**Early stopping en época 88.** El mejor F1-macro de validación se alcanzó en la **época 88**. Con una paciencia de 20 épocas, el entrenamiento completó las 100 épocas máximas (88 + 12 épocas restantes sin mejora en val hasta el límite), y los pesos se restauraron automáticamente a los de la época 88.

### 8.6 Importancia de features en XGBoost

XGBoost calcula la importancia de cada feature como la **ganancia media** aportada en las particiones del ensemble. Las 20 features de mayor importancia están disponibles en `output/pipeline_a/feature_importance.png`. Las cinco features más relevantes son:

| Ranking | Feature | Importancia | Categoría (§3.2) |
|---|---|---|---|
| 1 | `Subflow Bwd Bytes` | 0,134 | Tamaño del flujo |
| 2 | `Subflow Fwd Packets` | 0,119 | Tamaño del flujo |
| 3 | `Active Std` | 0,080 | Features de actividad |
| 4 | `Fwd PSH Flags` | 0,072 | Flags TCP |
| 5 | `Idle Mean` | 0,066 | Features de actividad |

Las dos features con mayor importancia (`Subflow Bwd Bytes`, `Subflow Fwd Packets`) cuantifican el volumen de datos e intercambio de paquetes en los subflujos de la conexión. Ataques DoS/DDoS generan subflujos con altísimo número de paquetes forward sin respuesta backward, mientras que ataques BruteForce producen subflujos pequeños y repetitivos con volumen backward reducido. Esta dualidad les otorga alto poder discriminativo en la partición multi-clase.

Las features de actividad e inactividad (`Active Std`, `Idle Mean`) aparecen en los puestos 3 y 5: el tráfico normal presenta variabilidad en los periodos de actividad y pausas de inactividad regulares entre ráfagas de datos, mientras que los ataques automatizados generan flujos continuos con poca variabilidad temporal (`Active Std` bajo) o períodos de inactividad anómalos. El `Fwd PSH Flag Count` discrimina eficazmente los ataques web (que usan PSH para enviar peticiones HTTP completas) del tráfico de escaneo (que genera mayoritariamente paquetes SYN/RST sin PSH).

Estos resultados son consistentes con el análisis EDA de §3.5: las features de subflow y de actividad son precisamente las que muestran mayor separabilidad por clase en los gráficos KDE.

---

## 9. Comparativa y análisis crítico de resultados

### 9.1 Tabla comparativa global

| Métrica | Pipeline A | Pipeline B | Δ (B − A) | Ganador |
|---|---|---|---|---|
| Accuracy | 0,9936 | 0,9685 | −0,0251 | Pipeline A |
| F1-macro | 0,9516 | 0,7497 | **−0,2020** | Pipeline A |
| F1-weighted | 0,9936 | 0,9736 | −0,0200 | Pipeline A |

Pipeline A gana en todas las métricas. La diferencia más significativa es el F1-macro, con una brecha de **20,2 puntos porcentuales**. Esta diferencia no se distribuye uniformemente entre las clases, sino que se concentra en las categorías de ataque más difíciles.

### 9.2 Análisis por clase

**Clases grandes — comportamiento comparable entre pipelines.** DDoS y DoS presentan resultados casi idénticos en ambos sistemas (F1 ≈ 0,98 y 0,97–0,98 respectivamente). Estas clases cuentan con miles de muestras en el test y sus patrones de flujo son suficientemente distintos del tráfico benigno para que ambos modelos los aprendan con facilidad. El gap entre pipelines en estas clases es inferior a 0,015 puntos, dentro del margen de variabilidad esperado.

**BruteForce — rendimiento alto en ambos pipelines.** Con F1 de 0,99 (A) y 0,96 (B), BruteForce es la clase minority mejor discriminada por ambos modelos. Los ataques de fuerza bruta presentan un patrón de red muy característico —múltiples intentos de autenticación fallidos en flujos cortos y repetitivos— que facilita la separación de BENIGN incluso con pocas muestras.

**Scan — primer gap significativo (Δ = −0,139).** Pipeline B obtiene recall = 1,00 en Scan pero precision = 0,74, lo que implica que la NN sobre-predice esta clase: asigna aproximadamente 1.900 muestras adicionales como Scan cuando en realidad pertenecen a otras categorías. XGBoost logra precision y recall del 0,99. Este patrón de sobre-predicción sugiere que la frontera de decisión aprendida para Scan por la NN es demasiado laxa, posiblemente porque en el espacio de features normalizado existen zonas de ambigüedad entre Scan y las clases minoritarias (Botnet, WebAttack) que el modelo resuelve asignando el valor de mayor frecuencia relativa en esa región.

**Botnet — brecha crítica (Δ = −0,606; F1: A = 0,76, B = 0,16).** Botnet representa el caso más extremo de fracaso de Pipeline B. Con precision = 0,08 y recall = 0,99, la NN detecta casi todos los flujos reales de botnet, pero genera aproximadamente **12 falsas alarmas por cada verdadero positivo**. Un sistema de alerta basado en Pipeline B para Botnet sería inoperante en producción. XGBoost, aunque también tiene dificultades con esta clase (precision = 0,62), mantiene un equilibrio que produce un F1 de 0,76, aceptable dado el soporte extremadamente bajo de la clase (116 muestras en test).

**WebAttack — brecha crítica análoga (Δ = −0,607; F1: A = 0,96, B = 0,35).** WebAttack en Pipeline B presenta el mismo patrón que Botnet: recall = 1,00 pero precision = 0,22 (aproximadamente 1 verdadero positivo por cada 3,5 falsas alarmas). XGBoost logra, en cambio, un F1 de 0,96, convirtiendo a Pipeline A en el único sistema viable para la detección operacional de ataques web con este dataset.

### 9.3 Diagnóstico del problema de precisión de la red neuronal

El patrón recurrente en Pipeline B —recall muy alto pero precision muy baja en las clases minoritarias extremas— apunta a un fallo específico del clasificador neuronal ante el desbalanceo severo, explicable por tres factores.

**Interacción del balanced batching y la Focal Loss.** El muestreo balanceado por batch amplifica artificialmente la representación de Botnet y WebAttack durante el entrenamiento: con solo 541 muestras de Botnet frente a 581.776 de BENIGN, el sobremuestreo con reemplazamiento hace que la red vea las mismas pocas muestras de Botnet repetidamente en múltiples batches. La red aprende a reconocer esas muestras específicas, pero sin la diversidad suficiente para desarrollar una frontera de decisión robusta frente a los falsos positivos del IF —flujos BENIGN que Stage 1 marcó como anómalos y que Stage 2 debe reclasificar—. La Focal Loss con $\gamma=2$ amplifica adicionalmente el gradiente de los ejemplos difíciles, lo que puede conducir a que la red priorice clasificar cualquier muestra similar a Botnet o WebAttack como tal, en detrimento de la precisión.

**Capacidad discriminativa frente al pool de falsos positivos del IF.** Con un FPR del 60 %, Stage 2 recibe aproximadamente 74.800 flujos BENIGN etiquetados como anómalos. Para la NN, estos flujos representan una fuente de ruido significativa: son muestras BENIGN que comparten suficientes características con flujos de ataque para haber superado el umbral del IF, y la NN debe aprender a discriminarlas de los ataques reales. XGBoost, con su capacidad discriminativa jerárquica de 300 árboles, gestiona este ruido de forma más eficiente que la NN.

**Limitaciones de la implementación sin framework.** XGBoost incorpora décadas de optimización de software (construcción de árboles por histogramas, regularización nativa L1/L2, early stopping integrado con validación), mientras que la NN NumPy implementa las mismas ideas matemáticamente correctas pero sin las optimizaciones de bajo nivel presentes en PyTorch o TensorFlow. Con GPU y un framework moderno, una arquitectura similar probablemente lograría mayor capacidad de generalización en clases ultraminoritarias.

### 9.4 Trade-off complejidad vs. rendimiento

El Pipeline A requirió aproximadamente **85 minutos** de ejecución en CPU, dominados por el RandomizedSearchCV con 20 iteraciones × 3 folds de validación cruzada. Pipeline B requirió **39 minutos**, incluyendo el ciclo de entrenamiento completo con early stopping. XGBoost es, por tanto, no solo más eficaz (+20 pp en F1-macro) sino también competitivo en tiempo al incluir el tuning de hiperparámetros. Sin búsqueda de hiperparámetros (`--skip-tuning`), el pipeline A entrenaría en pocos minutos.

Conviene cuantificar el coste real de la cascada frente a la promesa de eficiencia planteada en §1.3. Con un FPR del 60,08%, Stage 1 filtra directamente solo 49.765 de los 150.001 flujos de test (33,2%), enrutando el 66,3% restante a Stage 2. Esto está lejos del escenario ideal en que el filtro grueso descartaría el ~83% del tráfico (la proporción real de BENIGN) y reduciría el problema de Stage 2 a un subconjunto pequeño y enriquecido en ataques. En la práctica, la arquitectura en cascada tal como está calibrada (β=2,0) prioriza tan fuertemente el recall que renuncia a buena parte de la ventaja computacional que la motivaba, y Stage 2 termina procesando una fracción del tráfico comparable a la que procesaría un clasificador multi-clase aplicado directamente sobre los datos sin cascada.

Este resultado ilustra la dificultad de competir con métodos de boosting bien implementados usando una red neuronal de propósito general: la ventaja de la NN reside en su capacidad de aprender representaciones distribuidas complejas, pero esa ventaja solo se materializa con volúmenes de datos suficientes por clase, aceleración por GPU y optimizadores de alto nivel —condiciones que no se cumplen en el contexto de este proyecto.

### 9.5 Importancia de features en XGBoost

El análisis de importancia de features de XGBoost (disponible en `output/pipeline_a/feature_importance.png`) identifica las estadísticas de flujo más discriminativas para la clasificación multi-clase en Pipeline A. Las features de mayor importancia corresponden típicamente a estadísticas de duración del flujo, tasas de transferencia y longitudes de paquetes en la dirección forward, coherentes con el análisis EDA previo que mostró alta separabilidad de estas features por clase mediante el estadístico F de ANOVA.

---

## 10. Limitaciones del trabajo y posibles mejoras futuras

### 10.1 Limitaciones identificadas

**Alta tasa de falsos positivos del Isolation Forest.** El IF presenta un FPR del 60,08 %, muy por encima de lo que sería aceptable en un sistema operacional de producción. Esta característica es el resultado directo de la optimización con $\beta=2$, que prioriza deliberadamente el recall sobre la precisión. Si bien Stage 2 mitiga el efecto reclasificando la mayoría de los falsos positivos del IF como BENIGN, el volumen de flujos benignos que atraviesan Stage 2 es desproporcionadamente alto. Como consecuencia, en lugar de descartar en Stage 1 el 83 % del tráfico (la proporción real de BENIGN), el sistema solo descarta directamente el ~33 % (el 40 % de BENIGN que no supera el umbral del IF), enrutando el resto a la clasificación completa. Esto incrementa el tiempo de inferencia y la exposición de Stage 2 a ruido.

**Pipeline B por debajo del benchmark de F1-macro.** El F1-macro del Pipeline B (0,7497) no alcanza el objetivo de referencia de 0,80. La diferencia se concentra en Botnet y WebAttack, donde la precisión de la NN colapsa hasta el 8 % y el 22 % respectivamente. Este resultado refleja los límites de una red neuronal implementada en NumPy puro frente a herramientas como XGBoost, y pone de manifiesto que las técnicas de compensación del desbalanceo (balanced batching, Focal Loss, pesos de clase) no son suficientes para suplir la capacidad de un optimizador de árboles con regularización nativa cuando los datos de entrenamiento son escasos por clase.

**Entrenamiento en CPU sin aceleración por GPU.** La red neuronal del Pipeline B se ejecuta íntegramente en CPU mediante operaciones NumPy sobre matrices densas. La ausencia de aceleración GPU limita la capacidad de escalar a mayor número de parámetros, mayor número de épocas o técnicas de aumento de datos más intensivas. Con GPU, sería factible explorar arquitecturas más profundas o aplicar técnicas de *mixup* o SMOTE en el espacio de representación intermedio.

**Umbral único y global del Isolation Forest.** El IF utiliza un único umbral escalar aplicado a todos los flujos, independientemente de su perfil de ataque. Esto implica que el punto de operación es un compromiso global entre todos los tipos de ataque: clases con distribuciones de score muy distintas (por ejemplo, Botnet frente a DoS) no pueden beneficiarse de umbrales individualizados que optimicen su recall específico sin penalizar la precisión del sistema en otras categorías.

**Contaminacion temporal entre conjuntos**. Es importante señalar que la contaminación temporal entre conjuntos no es una posibilidad teórica sino una limitación documentada en la literatura sobre CICIDS2017: distintos trabajos han reportado que la partición aleatoria por fila —en lugar de por día de captura— infla las métricas de evaluación al permitir que flujos casi idénticos (mismo escenario de ataque, mismo host, variaciones mínimas de timing) aparezcan simultáneamente en train y test. La deduplicación exacta aplicada en este proyecto (§4.2) elimina copias idénticas pero no flujos casi-duplicados con pequeñas variaciones numéricas, que siguen pudiendo filtrar información entre particiones. En consecuencia, las métricas reportadas en §8 deben interpretarse como una cota optimista del rendimiento real del sistema ante tráfico genuinamente no visto, particularmente para las clases con escasas muestras (Botnet, WebAttack), donde la probabilidad de fuga de información entre train y test es proporcionalmente mayor al tratarse de un número reducido de escenarios de ataque repetidos. Una partición temporal estricta (por ejemplo, reservar los días viernes para test) garantizaría una evaluación más realista de la capacidad de generalización del sistema a tráfico futuro.

### 10.2 Posibles mejoras futuras

**Ensemble de los dos pipelines.** La combinación de las predicciones de Pipeline A y Pipeline B mediante votación suave (*soft voting*) o apilamiento (*stacking*) podría capturar el mejor rendimiento de cada sistema: XGBoost aporta precisión en las clases minoritarias, mientras que la NN puede capturar patrones más abstractos en clases con mayor volumen de datos. Dado que ambas arquitecturas modelan el espacio de features de formas radicalmente distintas, la diversidad del ensemble podría traducirse en mejoras sobre el F1-macro de cualquiera de los dos pipelines individuales.

**Técnicas de sobremuestreo sintético.** El muestreo balanceado por batch y la Focal Loss compensan el desbalanceo durante el entrenamiento, pero no añaden variabilidad a las clases minoritarias. SMOTE (*Synthetic Minority Over-sampling Technique*) o ADASYN generarían nuevas muestras sintéticas interpolando entre los vecinos más cercanos de las clases minoritarias, potencialmente mejorando la generalización de las fronteras de decisión para Botnet y WebAttack y reduciendo el problema de overfitting a las escasas muestras reales disponibles.

**Sustitución del Stage 1 por un clasificador binario supervisado.** El IF es un detector de anomalías no supervisado, entrenado sin información de etiqueta. Sustituirlo por un clasificador binario supervisado —por ejemplo, un XGBoost binario que distingue explícitamente BENIGN de "cualquier ataque"— podría reducir significativamente el FPR, aprovechando la información de clase disponible en el conjunto de entrenamiento. El riesgo de esta alternativa es la menor capacidad de generalización a ataques no vistos durante el entrenamiento, lo que habrá de equilibrarse experimentalmente.

**Selección y construcción de features.** El conjunto completo de 78 features incluye grupos con alta multicolinealidad (las features de longitud Forward presentan correlaciones de Pearson superiores a 0,9 entre sí, según el análisis EDA). Una selección basada en la importancia de XGBoost o en métodos *wrapper* podría reducir la dimensionalidad sin pérdida de rendimiento, aligerando el coste de Stage 2 y potencialmente mejorando la generalización de la NN al reducir la complejidad del espacio de entrada.

**Aprendizaje online e incrementalidad.** El dataset CICIDS2017 es estático, pero en producción el tráfico de red evoluciona continuamente. Un pipeline que incorpore aprendizaje online —por ejemplo, Hoeffding Trees para Stage 1 o un clasificador incremental para Stage 2— podría adaptarse a la deriva del concepto (*concept drift*) sin reentrenamiento completo. Esto es especialmente relevante para Botnet y WebAttack, cuyas herramientas y comportamientos de red evolucionan con rapidez y para las que un modelo estático tiende a degradarse con el tiempo.

---

## 11. Conclusiones

El objetivo principal del proyecto era diseñar y comparar experimentalmente dos pipelines de detección de intrusiones en cascada sobre CICIDS2017. Los resultados muestran una diferencia nítida entre ambos: **Pipeline A (IF + XGBoost) alcanzó un F1-macro de 0,9516**, superando el benchmark de referencia del 0,90 y demostrando una capacidad discriminativa sólida en todas las clases, incluidas las más minoritarias (WebAttack: F1 = 0,96; BruteForce: F1 = 0,99). **Pipeline B (IF + NN NumPy) obtuvo un F1-macro de 0,7497**, por debajo del objetivo de 0,80, con un colapso de precisión específico en Botnet (8 %) y WebAttack (22 %) que lo haría inoperante en producción para la detección de esas categorías. Estos resultados deben leerse con la cautela señalada en §10.1: la ausencia de partición temporal estricta implica que las cifras reportadas, especialmente en las clases minoritarias, probablemente sobrestiman el rendimiento esperable ante tráfico de ataque genuinamente nuevo.

El análisis por clase revela que la diferencia entre pipelines no es global sino concentrada en las clases con menos de 1.000 muestras de entrenamiento. En clases con suficiente representación (DDoS, DoS, Scan, BruteForce), ambos modelos rinden de forma comparable. La ventaja de XGBoost sobre la red neuronal NumPy se explica por tres factores combinados: mayor capacidad discriminativa jerárquica con 300 árboles regularizados, mayor robustez ante el ruido de los ~75.000 falsos positivos del IF que Stage 2 debe reclasificar, y décadas de optimización de software que la implementación NumPy no puede igualar sin GPU ni framework. En este sentido, **Pipeline A es el sistema recomendado** para cualquier despliegue operacional con el dataset CICIDS2017 tal como está configurado.

Más allá de las métricas, este proyecto ofreció dos aprendizajes fundamentales. Primero, que implementar manualmente la retropropagación, BatchNorm, Adam y Focal Loss en NumPy obliga a comprender los fundamentos matemáticos de las redes neuronales a un nivel que el uso de PyTorch o TensorFlow no requiere, y que ese esfuerzo tiene valor formativo independiente del resultado predictivo. Segundo, que en un sistema en cascada el cuello de botella irrecuperable es Stage 1: el FNR del 3,18 % del Isolation Forest impone un techo absoluto de recall del 96,82 % para todo el pipeline, independientemente de cuán bueno sea el clasificador de Stage 2. Esto subraya que en un IDS real la optimización del detector de anomalías inicial tiene un impacto mayor que cualquier mejora en el clasificador de la segunda etapa.
