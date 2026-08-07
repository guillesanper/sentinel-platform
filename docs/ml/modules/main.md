# Módulo `main`

## Descripción General

El módulo `main` constituye el punto de entrada principal del sistema de detección de intrusiones sobre el dataset CICIDS2017. Actúa como orquestador CLI que conecta de forma secuencial todos los subsistemas del proyecto: carga y preprocesamiento de datos, análisis exploratorio, partición train/val/test, entrenamiento y evaluación de los dos pipelines en cascada, y generación de figuras comparativas. La ejecución íntegra queda registrada en `output/results.txt` mediante el sistema de logging dual definido en `utils.logger`.

El diseño sigue el principio de separación de responsabilidades: `main.py` no contiene lógica de modelo ni de visualización propia; únicamente coordina llamadas a los módulos especializados definidos en las fases anteriores, garantizando que el experimento sea completamente reproducible dada la misma semilla aleatoria y los mismos datos de entrada.

## Fundamento Teórico

La reproducibilidad experimental exige fijar todas las fuentes de aleatoriedad del sistema antes de cualquier operación estocástica. En este módulo se fija la semilla de NumPy (`numpy.random.seed`) y del módulo `random` de Python estándar. Las semillas de scikit-learn y XGBoost se propagan mediante el parámetro `random_state` definido en `Config.RANDOM_SEED` y pasado a cada constructor de modelo.

La construcción de la matriz de probabilidades de cascada $\hat{P} \in [0,1]^{n \times K}$ (necesaria para las curvas ROC y PR de comparación) combina la predicción binaria del Isolation Forest con las probabilidades softmax del clasificador de Stage 2:

$$\hat{P}_{i,k} = \begin{cases} 1 & \text{si } \hat{y}^{\text{IF}}_i = \text{BENIGN} \text{ y } k = \text{BENIGN} \\ 0 & \text{si } \hat{y}^{\text{IF}}_i = \text{BENIGN} \text{ y } k \neq \text{BENIGN} \\ \hat{p}^{(2)}_{i, \sigma(k)} & \text{si } \hat{y}^{\text{IF}}_i = \text{anomalía} \end{cases}$$

donde $\sigma(k)$ es el índice de la clase $k$ en la codificación interna del clasificador de Stage 2, y $\hat{p}^{(2)}$ es la salida softmax de dicho clasificador.

## Funciones / Clases

### `parse_args() -> argparse.Namespace`
**Descripción:** Define y parsea la interfaz de línea de comandos completa del experimento.  
**Parámetros:** ninguno (lee `sys.argv`).  
**Retorna:** `argparse.Namespace` con los atributos:
- `data_path` — ruta al directorio con los CSVs de CICIDS2017
- `output_path` — directorio raíz de salida
- `sample_size` — tamaño del muestreo estratificado (0 = dataset completo)
- `seed` — semilla global
- `pipeline` — `"a"`, `"b"` o `"all"`
- `skip_eda` — booleano; omite la generación de figuras EDA
- `skip_tuning` — booleano; omite `RandomizedSearchCV` para XGBoost
- `beta` — parámetro F-beta para la optimización del umbral del Isolation Forest

**Notas:** Los valores por defecto replican exactamente la configuración estándar de `config.py` para garantizar la reproducibilidad sin flags adicionales.

---

### `_set_seeds(seed: int) -> None`
**Descripción:** Fija la semilla de NumPy y del módulo `random` de Python.  
**Parámetros:**
- `seed` — entero de semilla global

**Notas:** Las semillas de scikit-learn y XGBoost se propagan a través de `Config.RANDOM_SEED` en los constructores de cada modelo.

---

### `_build_config(args: argparse.Namespace) -> Config`
**Descripción:** Construye la instancia `Config` a partir de los argumentos CLI, sobreescribiendo los valores por defecto de la dataclass.  
**Parámetros:**
- `args` — namespace de argparse

**Retorna:** `Config` completamente inicializado con las rutas, semilla, tamaño de muestra y beta especificados.

---

### `_cascade_proba(if_model, stage2_predict_proba, stage2_classes, X, all_labels) -> np.ndarray`
**Descripción:** Construye la matriz de probabilidades $(n \times K)$ del pipeline en cascada para el cómputo de curvas ROC/PR multi-clase. Muestras no anómalas reciben probabilidad 1.0 en BENIGN; muestras anómalas reciben las probabilidades softmax del Stage-2 proyectadas sobre el índice canónico de clases.  
**Parámetros:**
- `if_model` — instancia entrenada de `IsolationForestModel`
- `stage2_predict_proba` — callable que acepta `X` y devuelve `(n, n_attack_classes)` probabilidades
- `stage2_classes` — array de nombres de clase en el orden interno del Stage-2
- `X` — matriz de características, shape `(n, n_features)`
- `all_labels` — lista canónica de $K$ clases

**Retorna:** `np.ndarray` shape `(n, K)`, dtype `float32`  
**Complejidad:** O(n · K)

---

### `_run_pipeline_a(...) -> Tuple[PipelineA, Dict]`
**Descripción:** Orquesta el ciclo completo del Pipeline A: entrenamiento (IF + XGBoost), evaluación end-to-end en test, persistencia del modelo, generación de cuatro figuras (matriz de confusión, importancia de features, curva de umbral IF y curva precision-recall del detector binario) y emisión del `classification_report` de scikit-learn con precision/recall/F1 por clase en el log.  
**Parámetros:**
- `X_train, y_train` — conjunto de entrenamiento completo
- `X_val, y_val` — conjunto de validación
- `X_test, y_test` — conjunto de test (evaluación final)
- `feature_names` — nombres de columna de features para `get_feature_importance`
- `all_labels` — clases canónicas activas en el experimento
- `config` — instancia `Config`
- `args` — namespace CLI (para `skip_tuning` y `beta`)

**Retorna:** tupla `(PipelineA entrenado, dict de métricas de test)`

---

### `_run_pipeline_b(...) -> Tuple[PipelineB, Dict]`
**Descripción:** Análogo a `_run_pipeline_a` para el Pipeline B (IF + NumPy MLP). Genera matriz de confusión, curvas de entrenamiento (loss/accuracy por epoch con marcador de early stopping), curva de umbral IF y curva precision-recall del detector binario, y emite el `classification_report` con precision/recall/F1 por clase en el log.  
**Parámetros:** ídem a `_run_pipeline_a`, sin `feature_names` (la red neuronal no expone importancia de variables).  
**Retorna:** tupla `(PipelineB entrenado, dict de métricas de test)`

---

### `_run_comparison(pa, pb, X_test, y_test, all_labels, config) -> None`
**Descripción:** Ejecuta la comparativa cuantitativa entre ambos pipelines sobre el conjunto de test. Construye primero las matrices de probabilidades de cascada mediante `_cascade_proba` y las pasa a `compare_pipelines` para habilitar el cómputo de AUC-ROC en la tabla de comparación. A continuación registra los indicadores del Stage-1 (Isolation Forest) de ambos pipelines: precision, recall, FPR y FNR del detector binario. Finalmente genera cuatro figuras de comparación: métricas globales agrupadas, F1 por clase, curvas ROC OvR y curvas PR OvR.  
**Parámetros:**
- `pa` — `PipelineA` entrenado
- `pb` — `PipelineB` entrenado
- `X_test, y_test` — conjunto de test
- `all_labels` — orden canónico de clases
- `config` — instancia `Config`

**Notas:** Si la construcción de las matrices de probabilidad falla, `compare_pipelines` se invoca sin scores (sin AUC-ROC) y las curvas ROC/PR se omiten con `WARNING`. Los indicadores IF se omiten igualmente con `WARNING` si el método `get_if_operating_point` no está disponible.

---

### `main() -> None`
**Descripción:** Función principal. Secuencia de ejecución:
1. Parseo de argumentos CLI.
2. Construcción de `Config` y creación de directorios de salida.
3. Inicialización del logger dual (consola + `output/results.txt`).
4. Fijación de semillas globales.
5. Carga (`load_cicids2017`), limpieza (`clean_dataset`), filtrado de clases minoritarias (`filter_minority_classes`) y muestreo estratificado (`stratified_sample`).
6. EDA opcional (`run_eda`).
7. Partición train/val/test (`split_dataset`) y extracción de arrays NumPy.
8. Ejecución condicional de `_run_pipeline_a` y/o `_run_pipeline_b`.
9. Comparativa cruzada opcional (`_run_comparison`).
10. Bloque de resumen final con F1-macro, F1-weighted y accuracy de cada pipeline activo.
11. Log de tiempo total transcurrido.

**Parámetros:** ninguno (lee `sys.argv` vía `parse_args`).  
**Retorna:** ninguno. Efectos secundarios: archivos en `output/`.

## Flujo de Control

```
main()
  ├── parse_args()
  ├── _build_config()
  ├── config.create_output_dirs()
  ├── setup_logger()
  ├── _set_seeds()
  │
  ├── load_cicids2017()
  ├── clean_dataset()
  ├── filter_minority_classes()
  ├── stratified_sample()
  ├── [run_eda()]  ← skip con --skip-eda
  ├── split_dataset()
  │
  ├── [_run_pipeline_a()]  ← si --pipeline in {a, all}
  │     ├── PipelineA.train()
  │     ├── PipelineA.evaluate()
  │     ├── PipelineA.save()
  │     └── plot_{confusion_matrix, feature_importance,
  │                threshold_curve, pr_tradeoff}()
  │
  ├── [_run_pipeline_b()]  ← si --pipeline in {b, all}
  │     ├── PipelineB.train()
  │     ├── PipelineB.evaluate()
  │     ├── PipelineB.save()
  │     └── plot_{confusion_matrix, training_curves,
  │                threshold_curve, pr_tradeoff}()
  │
  ├── [_run_comparison()]  ← si ambos pipelines activos
  │     ├── _cascade_proba()  ×2  ← matrices de probabilidad para AUC-ROC
  │     ├── compare_pipelines()   ← con y_score_a / y_score_b
  │     ├── log IF operating point (A y B)
  │     └── plot_{metrics_bar, f1_per_class, roc_curves, pr_curves}()
  │
  └── log RESUMEN FINAL DE MÉTRICAS
```

## Especificación de la Interfaz CLI

```
usage: main.py [--data-path PATH] [--output-path PATH]
               [--sample-size N] [--seed N]
               [--pipeline {a,b,all}]
               [--skip-eda] [--skip-tuning]
               [--beta FLOAT]
```

| Flag | Tipo | Defecto | Descripción |
|---|---|---|---|
| `--data-path` | `Path` | `data/` | Directorio con CSVs de CICIDS2017 |
| `--output-path` | `Path` | `output/` | Raíz de salida del experimento |
| `--sample-size` | `int` | `200000` | Muestras estratificadas (0 = total) |
| `--seed` | `int` | `42` | Semilla global de aleatoriedad |
| `--pipeline` | `{a,b,all}` | `all` | Pipeline(s) a ejecutar |
| `--skip-eda` | flag | `False` | Omite figuras EDA |
| `--skip-tuning` | flag | `False` | Omite búsqueda de hiperparámetros XGBoost |
| `--beta` | `float` | `1.0` | F-beta para umbral IF (>1 → más recall; <1 → más precisión) |

## Dependencias Internas

- `config` — `Config`, `BENIGN_LABEL`, `CLASS_ORDER`
- `data.loader` — `load_cicids2017`, `clean_dataset`, `filter_minority_classes`, `stratified_sample`, `run_eda`, `split_dataset`, `LABEL_COL`
- `evaluation.metrics` — `compare_pipelines`
- `evaluation.visualization` — todas las funciones de visualización
- `pipelines.pipeline_a` — `PipelineA`
- `pipelines.pipeline_b` — `PipelineB`
- `utils.logger` — `setup_logger`, `get_logger`

## Referencias

- Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*, 108–116.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.
- Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*, 233–240.
