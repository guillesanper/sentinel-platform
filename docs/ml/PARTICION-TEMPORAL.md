# La partición temporal del track de flujos: qué medía y qué mide ahora

Este documento registra un error de concepto que llegó a estar en la rama, cómo se detectó y cómo
se corrigió. Se conserva porque la afirmación equivocada era **la afirmación estrella del track de
flujos**, y porque el mecanismo que la produjo es fácil de repetir en cualquier dataset que colapse
etiquetas antes de partir por tiempo.

Todas las cifras de este documento se re-midieron el **22 de agosto de 2026** sobre los CSV reales
de CICIDS2017 en `api/csvs/` (2.520.798 filas tras limpieza), con:

```powershell
python -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 `
  --pipeline xgb --skip-tuning --no-plots --no-mlflow --split <protocolo>
```

> **Las cifras cambiaron respecto a la primera versión y el motivo es un tercer error, de la misma
> familia que los dos que este documento registra.** `order_idx` se derivaba del orden en que el
> loader leía los ficheros, que era el alfabético, y ése **no es el orden de captura**. El registro
> está en la [§11](#11-corolario-2-22-de-agosto-de-2026-order_idx-no-era-cronológico-entre-ficheros),
> con el experimento de control que reproduce exactamente las cifras publicadas antes. Los §§1-2
> conservan las cifras de agosto porque son el objeto del registro histórico; las tablas de
> resultados de los §§3-5 y §10 son las del código actual.

---

## 1. Qué se afirmaba

La documentación del track presentaba esta comparación:

| Split | Accuracy | F1-macro |
|---|---|---|
| `random` | 0.9934 | **0.9166** |
| `temporal` | 0.8783 | **0.2809** |

y la leía así:

> **La caída de 0.9166 a 0.2809 en F1-macro no es una regresión: es la magnitud de la fuga.**

La conclusión operativa era que el 0.95 publicado en la memoria del TFA estaba inflado por
casi-duplicados de la misma ráfaga repartidos entre train y test, y que el rendimiento honesto del
diseño rondaba el 0.28. Es una afirmación fuerte y era el resultado más citado de la rama.

**Es falsa por un factor de ~4.** La fuga existe y se midió mal: 0.148 de los 0.622 que caían son
fuga; los otros 0.475 son un artefacto de cómo se definía un bloque temporal.

---

## 2. Por qué es falsa

`split_dataset_temporal` corta cronológicamente **dentro de cada bloque**, y hasta ahora un bloque
era el par `(capture_day, clase canónica)`. El problema está en "clase canónica":
[`ATTACK_GROUPS`](../../ml/flow/labels.py) colapsa 15 etiquetas crudas en 8 clases, y **las
herramientas de ataque que se colapsan juntas corrieron una detrás de otra dentro del mismo día de
captura**.

Cortar por posición dentro de un bloque así no separa "antes" de "después" dentro de un escenario:
separa **una herramienta de otra**. Composición real de cada bloque de ataque bajo el corte
70/15/15 (dataset completo, sin muestrear):

| Bloque | Herramientas | train (primer 70 %) | test (último 15 %) |
|---|---|---|---|
| `wednesday/DoS` | 5 | Hulk 125.016 · slowloris 5.385 · Slowhttptest 5.228 | Hulk 18.767 · **GoldenEye 10.286** · **Heartbleed 11** |
| `thursday/WebAttack` | 3 | Brute Force 1.470 · XSS 30 | **XSS 301** · **Sql Injection 21** |
| `tuesday/BruteForce` | 2 | FTP-Patator 5.931 · SSH-Patator 474 | **SSH-Patator 1.373 (100 %)** |
| `friday/DDoS` | 1 | DDoS 89.609 | DDoS 19.203 |
| `friday/Scan` | 1 | PortScan 63.485 | PortScan 13.605 |
| `friday/Botnet` | 1 | Bot 1.363 | Bot 293 |
| `thursday/Infiltration` | 1 | Infiltration 25 | Infiltration 6 |

En las tres primeras filas el test no mide lo mismo que el train:

* **`wednesday/DoS`**: el 35,4 % del test son GoldenEye y Heartbleed, que **no aparecen ni una vez**
  en train.
* **`thursday/WebAttack`**: Sql Injection no aparece en train, y XSS aparece 30 veces contra 301 en
  test — el clasificador ve el 9 % de la evidencia que se le exige reproducir.
* **`tuesday/BruteForce`**: el test es **íntegramente** SSH-Patator, contra 474 filas de
  SSH-Patator en un train de 6.405.

Eso no es una partición anti-fuga. Es un **holdout de herramienta de ataque no vista**, disfrazado
de partición anti-fuga. Es una pregunta legítima y difícil, pero es **otra pregunta**, y nadie la
había elegido.

---

## 3. El experimento que separa las dos cosas

Un tercer protocolo, con **exactamente la misma garantía cronológica**, cambiando solo la clave del
bloque: `(capture_day, etiqueta cruda)` en lugar de `(capture_day, clase canónica)`. Cada
herramienta se corta sobre su propia línea temporal, así que ninguna queda fuera de train y ningún
flujo de test precede a uno de train dentro de su escenario.

| Protocolo | Accuracy | F1-macro | F1-weighted | Stage 1 P / R / FPR (test) |
|---|---|---|---|---|
| `--split random` | 0.9944 | **0.9453** | 0.9945 | 0.250 / 0.976 / 0.596 |
| `--split temporal --temporal-block sub-label` *(default)* | 0.9194 | **0.6467** | 0.9052 | 0.335 / 0.560 / 0.226 |
| `--split temporal --temporal-block class` | 0.9165 | **0.6547** | 0.8999 | 0.329 / 0.573 / 0.237 |

> La columna de Stage 1 está medida sobre **test**. En la primera versión de este documento traía
> las cifras de **validación**, que es donde se *elige* el umbral; ver §10.

> **Estas tres filas no son las publicadas en agosto** (0.9197 / 0.7721 / 0.2973). Las de entonces
> se midieron con `order_idx` derivado del orden alfabético de los ficheros; con el orden de captura
> real, el umbral de Stage 1 aterriza en otro sitio y arrastra las tres. El experimento de control
> —código actual, orden alfabético restaurado— reproduce las cifras de agosto **exactamente**, así
> que el movimiento tiene una sola causa y está aislada. Ver §11.

F1 por clase (`compute_metrics` sobre la lista canónica de clases), con el número de herramientas
que compone cada clase y su soporte en test:

| Clase | Herram. | Soporte | `random` | `sub-label` | `class` |
|---|---|---|---|---|---|
| BENIGN | — | 24.936 | 0.997 | 0.954 | 0.952 |
| Botnet | 1 | 24 | 0.708 | **0.000** | **0.000** |
| BruteForce | 2 | 110 | 0.968 | 0.934 | 0.969 |
| DDoS | 1 | 1.524 | 0.993 | 0.886 | 0.893 |
| DoS | 5 | 2.308 | 0.976 | 0.732 | 0.653 |
| Scan | 1 | 1.080 | 0.995 | **0.201** | **0.244** |
| WebAttack | 3 | 26 | 0.981 | 0.820 | 0.873 |

Y el **recall de Stage 1 por clase**, con el umbral congelado y medido sobre test. Es la tabla que
falta en casi cualquier informe de cascada, y sin ella la de arriba no se puede leer: un F1 de
Stage 2 solo puede medirse sobre las filas que Stage 1 dejó pasar.

| Clase | `random` | `sub-label` | `class` |
|---|---|---|---|
| BENIGN *(es FPR, no recall)* | 0.596 | 0.226 | 0.237 |
| Botnet | 1.000 | **0.000** | **0.000** |
| BruteForce | 0.982 | 0.973 | 0.991 |
| DDoS | 0.990 | 0.796 | 0.806 |
| DoS | 0.956 | 0.594 | 0.603 |
| Scan | 0.999 | **0.113** | **0.140** |
| WebAttack | 1.000 | 0.962 | 1.000 |

Las dos tablas juntas dicen algo que ninguna dice sola: bajo los protocolos temporales, el F1 de
`Scan` (0.20) y el de `Botnet` (0.00) **no son veredictos sobre Stage 2**. Stage 1 le entrega 122 de
1.080 flujos de Scan y ninguno de los 24 de Botnet. Lo que miden es el techo, no el clasificador.

---

## 4. La descomposición

```
7 de agosto de 2026 (order_idx alfabético)
F1-macro   random 0.9197  →  sub-label 0.7721  →  class 0.2973
           └──── 0.1476 ────┘└──────── 0.4748 ────────┘
              fuga (23,7 %)    artefacto taxonomía × tiempo (76,3 %)

22 de agosto de 2026, con order_idx en orden de captura (§11)
F1-macro   random 0.9453  →  sub-label 0.6467  →  class 0.6547
           └──── 0.2986 ────┘└──────── −0.0080 ────────┘
            protocolo (100 %)   artefacto taxonomía × tiempo (no medible)
```

**La descomposición de agosto no sobrevive a la re-medición, y el motivo lo explica este mismo
documento tres párrafos más abajo.** El artefacto de taxonomía nunca fue un efecto directo sobre
Stage 2: su vía era el **punto de operación compartido de Stage 1**, y con el orden de captura
corregido los dos protocolos aterrizan en umbrales casi iguales (0.0655 bajo `sub-label`, 0.0747
bajo `class`; recall de Stage 1 0.560 y 0.573). Cuando el mediador deja de moverse, el efecto
desaparece: `class` queda 0.008 **por encima** de `sub-label`, dentro de lo que separa dos
protocolos que difieren en cómo se cortan tres bloques de siete.

Lo que sigue en pie de la §2 es un hecho del dataset, no una métrica: bajo `class` el test de
`tuesday/BruteForce` **sigue siendo 100 % SSH-Patator** y el 35 % del de `wednesday/DoS` sigue
siendo GoldenEye y Heartbleed. Lo que ya no se sostiene es que eso costara 0.475 de F1-macro. Con
las cifras actuales, `BruteForce` puntúa **mejor** bajo `class` (0.969) que bajo `sub-label`
(0.934) — es decir, la clase de dos herramientas nombra correctamente la herramienta que no vio.

**Lo que queda es la diferencia entre el protocolo aleatorio y el temporal: 0.2986 de F1-macro.**
Llamarla «fuga por casi-duplicados» sin más sería repetir el error de la §1 con otro número: la
tabla de recall de Stage 1 por clase de la §3 muestra que buena parte de esa diferencia se paga
**antes** de que Stage 2 opine. Bajo split aleatorio Stage 1 marca ≥ 0.95 de cada clase de ataque;
bajo el temporal marca 0.11 de `Scan` y 0.00 de `Botnet`. La descomposición honesta de esos 0.2986
es «casi-duplicados que Stage 2 memorizaba» **más** «un umbral de Stage 1 que no transfiere en el
tiempo», y este documento no las separa. La §5 mide la segunda con el suelo de muestreo.

### Por qué el F1-macro del protocolo `class` no es un agregado coherente

Aunque midiera lo que se pretendía, el número seguiría sin poder promediarse. El protocolo `class`
solo exige generalizar a una herramienta nueva **en las clases que tienen más de una herramienta**:

* `BruteForce`, `DoS` y `WebAttack` responden a "¿generaliza a una herramienta no vista?".
* `Botnet`, `DDoS` y `Scan` tienen una sola herramienta, así que su bloque se parte **exactamente
  igual** bajo los dos protocolos y responden a "¿detecta lo que ya vio?".

Promediar las seis da un número que no responde a ninguna de las dos preguntas. Es la razón de
fondo por la que el protocolo se conserva pero **se reporta aparte, nunca como titular**.

### El artefacto no se queda en las clases multi-herramienta

Es el hallazgo que no esperábamos y el que justificó el cambio de default. `Scan` y `DDoS`
tienen **una sola herramienta**: su partición es idéntica bit a bit bajo `sub-label` y bajo `class`.
Y sin embargo, en la medición de agosto, caían de 0.80 a 0.01 y de 0.98 a 0.63 al cambiar la clave
de bloqueo. (Re-medidas, ya no: 0.201 → 0.244 y 0.886 → 0.893. El hallazgo era correcto y su
mecanismo también; lo que ha desaparecido es la magnitud, por lo que explica la §11.)

La vía es el **punto de operación compartido de Stage 1**. El umbral del Isolation Forest se elige
maximizando F-beta sobre validación, y la composición de validación la fija la misma clave de
bloqueo. Bajo `class`, la búsqueda aterriza en un umbral radicalmente más conservador:

| Protocolo | Stage 1 recall (test) | Stage 1 FPR (test) | |
|---|---|---|---|
| `sub-label` | 0.804 | 0.491 | *agosto, orden alfabético* |
| `class` | **0.393** | 0.038 | *agosto, orden alfabético* |
| `sub-label` | 0.560 | 0.226 | *re-medido, orden de captura* |
| `class` | 0.573 | 0.237 | *re-medido, orden de captura* |

En agosto, con `class`, **el 61 % de los ataques no superaba Stage 1 y por tanto nunca llegaba a
Stage 2**; las clases de una sola herramienta pagaban la factura igual que las demás. Re-medido, el
umbral ya no se dispara al cambiar la clave del bloque, y por eso el artefacto se evapora.

El mecanismo que este párrafo describía sigue siendo correcto y ahora tiene una segunda
confirmación: **la cascada tiene un único umbral para las siete clases, y cualquier cosa que mueva
la composición de validación mueve las métricas de todas.** En agosto lo movía la clave del bloque;
en la re-medición lo mueve el orden de los ficheros (§11) y lo mueve el suelo de muestreo (§5). Tres
causas distintas, un solo canal.

---

## 5. Botnet: el caso donde el veredicto duro está entero

`Botnet` cae **0.71 → 0.00 en los dos protocolos temporales**.

Tiene una sola sub-etiqueta (`Bot`), un solo día y 1.948 filas tras limpieza, así que aquí no hay
confusión posible con la novedad: su bloque se parte idénticamente bajo ambas claves y aun así el
modelo pasa de 0.71 a cero en cuanto los casi-duplicados dejan de cruzar la frontera train/test.
**Ese 0.71 era memorización pura.**

Hasta esta revisión la explicación se detenía en el soporte: con la muestra de 200.000 la clase se
queda en **108 filas de entrenamiento y 24 de test**, y sobre 24 filas 0.00 y 0.50 son la misma
medida. Esa objeción era legítima y ya no aplica, porque ahora se puede desactivar.

### El suelo de muestreo, y lo que contesta

`--class-sample-floor 2000` cambia el reparto de la muestra: en vez de estrictamente proporcional,
cada clase se lleva al menos el suelo o todas las filas que tiene. Botnet pasa de 155 filas a las
1.948 que existen — **1.363 de entrenamiento y 293 de test, 12,6× más que antes**. El resto del
presupuesto se reparte proporcionalmente, así que el total sigue siendo exactamente 200.000.

| Run | Botnet train | Botnet test | Recall de Stage 1 en Botnet | **F1 de Stage 2 en Botnet** |
|---|---|---|---|---|
| `sub-label`, sin suelo, orden alfabético *(lo publicado en agosto)* | 108 | 24 | 1.000 (24 de 24) | 0.000 |
| `sub-label`, sin suelo | 108 | 24 | **0.000** (0 de 24) | 0.000 |
| `sub-label`, suelo 2.000 | 1.363 | 293 | **1.000** (293 de 293) | **0.000** |
| `class`, suelo 2.000 | 1.363 | 293 | 1.000 | **0.000** |
| `random`, suelo 2.000 | 1.364 | 292 | 1.000 | 0.973 |

**La respuesta no cambia.** Con todas las filas que la clase tiene, con Stage 1 entregándole el
100 % del test y con 1.363 filas de entrenamiento, Stage 2 sigue en 0.00 bajo los dos protocolos
temporales — y en 0.97 bajo el aleatorio, con el mismo volumen de datos y sobre el mismo muestreo.
Lo que falta no es soporte: es que un botnet capturado **un solo día con una sola herramienta**,
cortado cronológicamente, no deja nada transferible del primer 70 % al último 15 %.

La primera fila de la tabla añade el matiz que impide una lectura cómoda del arreglo del orden de
captura: **bajo las cifras publicadas en agosto Stage 1 sí marcaba las 24 filas de Botnet, y el F1
era 0.00 igualmente.** Que hoy no las marque empeora el diagnóstico, no lo cambia.

El suelo, por tanto, no arregla Botnet. Lo que hace es **retirar la única objeción metodológica al
0.00** y dejar el veredicto sin coartada.

### La trampa narrativa, que aquí es literal

El suelo **sí mueve el titular**: F1-macro `sub-label` pasa de 0.6467 a **0.7893**. Sería fácil
presentarlo como «arreglamos las clases raras». No es eso lo que pasó:

| Clase | train sin suelo → con suelo | F1 sin suelo → con suelo | S1 recall sin suelo → con suelo |
|---|---|---|---|
| Botnet | 108 → 1.363 | 0.000 → **0.000** | 0.000 → 1.000 |
| Scan | 5.037 → **4.913** | 0.201 → **0.792** | 0.113 → 0.667 |
| DDoS | 7.109 → **6.934** | 0.886 → 0.956 | 0.796 → 0.921 |
| DoS | 10.758 → **10.495** | 0.732 → 0.847 | 0.594 → 0.754 |
| WebAttack | 118 → 1.398 | 0.820 → 0.971 | 0.962 → 0.993 |
| BruteForce | 507 → 1.399 | 0.934 → 0.987 | 0.973 → 0.990 |

**`Scan`, `DDoS` y `DoS` mejoran con menos filas de entrenamiento que antes.** No aprendieron nada
nuevo: lo que cambió es que las 4.897 filas de ataque que el suelo añade —el 2,45 % de la
muestra, restadas sobre todo a BENIGN (−4.092)— mueven el ratio
BENIGN:ataque que ve la búsqueda F-beta, el umbral de Stage 1 sube de 0.0655 a 0.0885, y con él el
recall de Stage 1 de 0.560 a 0.819. Casi toda la ganancia del titular es eso. La clase para la que
se construyó el suelo es exactamente la que no se mueve.

Es el acoplamiento de la §4 una vez más, y la razón de que el suelo venga **desactivado por
defecto**: no es una opción de muestreo, es un cambio del punto de operación disfrazado de opción
de muestreo.

### Lo que quedaría por intentar, y por qué no está aquí

Las dos vías que atacarían lo que queda de Botnet están **fuera de alcance** y conviene decir cuáles
son para que nadie las re-derive: **umbrales de Stage 1 por clase** en lugar de uno global
(`memoria-cicids2017.md:700`) —es un rediseño de la cascada, no un parámetro— y **migrar a
`GeneratedLabelledFlows`** (`CICIDS2017-TRACK.md`, *Trabajo futuro*), que traería `Timestamp` real
y agrupación por host. Ni siquiera la segunda arregla Botnet: seguiría siendo un día y una
herramienta.

---

## 6. Qué protocolo responde a qué pregunta

| Pregunta | Protocolo | Comando |
|---|---|---|
| ¿Cuánto detecta de verdad, sin fuga por casi-duplicados? | **`temporal` + `sub-label`** (default) | `--split temporal` |
| ¿Cuánto de lo publicado era memorización? | Contraste `random` vs `sub-label` | `--split random` |
| ¿Generaliza a una herramienta de ataque no vista? | `temporal` + `class` | `--split temporal --temporal-block class` |
| ¿Generaliza a un día no visto? | `day-holdout` | `--split day-holdout --test-days friday` |

El default es `sub-label` porque es el que responde a la pregunta que la plataforma necesita
contestar antes de desplegar nada: *cuánto detecta el modelo sobre tráfico que no ha visto, medido
sin fuga*. El protocolo `class` sigue disponible, se ejecuta con una sola opción y **su resultado
sigue publicado en este documento**: no se ha borrado un mal número, se ha etiquetado.

> **El cambio de default movió el número titular de la rama: 0.2809 → 0.7721.** Es intencionado y
> es el objeto del cambio. La cifra que sustituye no era más favorable por casualidad: es la que
> corresponde a la pregunta que se creía estar haciendo desde el principio.
>
> **Enmienda del 22 de agosto de 2026:** re-medido con `order_idx` en orden de captura, el cambio de
> default ya casi no mueve el titular (0.6547 bajo `class` contra 0.6467 bajo `sub-label`). El
> argumento para el default **no cambia** —sigue siendo el único de los dos que responde a la
> pregunta que la plataforma necesita contestar— pero ya no se apoya en una diferencia de 0.47 de
> F1-macro, porque esa diferencia era en su mayor parte un artefacto del orden de los ficheros. Un
> default que solo se justificaba por su número habría quedado sin justificación; éste no.

---

## 7. El arreglo, fichero por fichero

| Fichero | Cambio |
|---|---|
| [`ml/flow/loader.py`](../../ml/flow/loader.py) | Nueva columna de procedencia `raw_label`, poblada **antes** de que el mapeo canónico sobrescriba `Label`, y añadida a `META_COLUMNS` para que no pueda llegar al modelo. `_resolve_block_columns` elige la clave del bloque; `_temporal_cut` la recibe como parámetro. `clean_dataset` deduplica incluyendo `raw_label`. `_log_split_distribution` emite además el reparto por sub-etiqueta |
| [`ml/flow/config.py`](../../ml/flow/config.py) | `TEMPORAL_BLOCKS = ("sub-label", "class")` y el campo `temporal_block`, con default `"sub-label"` y validación en `__post_init__` |
| [`ml/flow/labels.py`](../../ml/flow/labels.py) | `normalize_raw_label()` — la reparación de mojibake **sin** el colapso. `Infiltration` deja de ser `Scan` (ver abajo) |
| [`ml/training/train_cicids.py`](../../ml/training/train_cicids.py) | Flag `--temporal-block`; el bloque `split` del `training_summary.json` registra la clave usada (`null` bajo `--split random`, que no tiene bloques) |
| [`tests/unit/test_flow_loader.py`](../../tests/unit/test_flow_loader.py) | `TestTemporalBlockKey` fija el hallazgo: con `sub-label` cada herramienta está en train, val y test; con `class` el test de una clase multi-herramienta se compone de una herramienta ausente de train |

### La segunda tanda (22 de agosto de 2026)

Tres deudas que el arreglo anterior dejó abiertas, más el hallazgo que las acompañó.

| Fichero | Cambio |
|---|---|
| [`ml/flow/cv.py`](../../ml/flow/cv.py) *(nuevo)* | `TemporalIndex` y `temporal_block_folds`: folds de **ventana expansiva por bloque** para el tuning de Stage 2. Con `cv_folds=K` cada bloque `(capture_day, raw_label)` se corta en `K+1` trozos y el fold *k* entrena en `0..k-1` y valida en el *k*. Corta con `loader.block_boundaries`, promovido a público para que exista una sola implementación de la regla. Ver §12 |
| [`ml/flow/models/xgboost_classifier.py`](../../ml/flow/models/xgboost_classifier.py) | `tune_hyperparameters(..., *, train_index=None)` usa esos folds en lugar de `StratifiedKFold(shuffle=True)`, con fallback avisado. El scorer pasa de la cadena `"f1_macro"` a `make_scorer(..., labels=np.arange(n_classes))` — ver §12, es el mismo defecto que el titular sin `labels=` |
| [`ml/flow/labels.py`](../../ml/flow/labels.py) | `CAPTURE_FILE_ORDER` y `capture_file_rank()`: el orden real de captura de los ocho ficheros, que **no** es el alfabético. Ver §11 |
| [`ml/flow/loader.py`](../../ml/flow/loader.py) | Los CSV se leen por `(capture_file_rank, nombre)`, así que `order_idx` es cronológico también **entre** ficheros; el orden resuelto se emite en `cicids_load_started`. Nuevo camino de muestreo con suelo (`_floored_sample`, `_class_quotas`), `audit_class_support()` tras la partición, y `_SMALL_BLOCK_ROWS` promovido a `MIN_MEANINGFUL_BLOCK_ROWS` |
| [`ml/flow/pipelines/base.py`](../../ml/flow/pipelines/base.py) y las dos cascadas | `evaluate(..., labels=)`: el `f1_macro` titular se promedia sobre la lista canónica de clases y no sobre `y_test ∪ y_pred`, que es lo que hacía desaparecer del macro a una clase sin soporte en test. `train(..., train_index=)` transporta el índice temporal a Stage 2 |
| [`ml/flow/config.py`](../../ml/flow/config.py) | `class_sample_floor: int = 0`, validado `>= 0`. Opt-in **a propósito**: con el default, los runs publicados siguen siendo reproducibles (§5) |
| [`ml/training/train_cicids.py`](../../ml/training/train_cicids.py) | Flag `--class-sample-floor`; el summary gana `per_class` (F1 y soporte por clase de cada cascada) y `class_support` (filas por clase y split, de la auditoría) |
| [`tests/unit/test_flow_cv.py`](../../tests/unit/test_flow_cv.py) *(nuevo)* | `TestTemporalBlockFolds`. El test que fija la trampa es el de que los folds devuelven **posiciones de fila y no valores de `order_idx`**: `_temporal_cut` baraja cada parte al final, así que el orden de filas del `train_df` no es cronológico aunque el valor de `order_idx` sí lo sea |

### `Infiltration` sale de `Scan`

Cambio adyacente, encontrado en la misma revisión. `ATTACK_GROUPS` mapeaba `Infiltration → Scan`,
metiendo 36 flujos de un escenario de compromiso de host y exfiltración dentro de 90.694 de barrido
de puertos. Además **eludía el filtro de minoritarias**: `filter_minority_classes` corre *después*
del colapso, así que `min_class_samples=100` nunca llegaba a ver esas 36 filas.

Con entrada propia en `CLASS_ORDER`, el filtro la descarta por sí solo — que es exactamente para lo
que está — y lo deja registrado en el log del run (`minority_class_removed`). El efecto sobre las
métricas es despreciable (36 filas de 2,5 M); el efecto sobre la corrección es que desaparece un
bloque `thursday/Scan` que aportaba un test garantizado a 0.

---

## 8. Cómo reproducirlo

```powershell
# Los tres protocolos, ~85 s de run completo cada uno, carga de CSV incluida,
# con los datos reales en api/csvs/
$py = "python"
& $py -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 --pipeline xgb `
    --skip-tuning --no-plots --no-mlflow --split random         --output-dir "$env:TEMP\flow-random"
& $py -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 --pipeline xgb `
    --skip-tuning --no-plots --no-mlflow --temporal-block class --output-dir "$env:TEMP\flow-coarse"
& $py -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 --pipeline xgb `
    --skip-tuning --no-plots --no-mlflow                        --output-dir models/flow

# El delta del suelo de muestreo (§5). Mismos tres protocolos, con 2.000 filas
# mínimas por clase. NO es la línea base: mueve el punto de operación.
& $py -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 --pipeline xgb `
    --skip-tuning --no-plots --no-mlflow --class-sample-floor 2000 `
    --output-dir "$env:TEMP\flow-floor"

# Con la búsqueda de hiperparámetros de Stage 2 sobre folds cronológicos (§12).
# 602 s de reloj de pared medidos, contra los 10.800 de activeDeadlineSeconds.
& $py -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 --pipeline xgb `
    --tune --no-plots --no-mlflow --output-dir "$env:TEMP\flow-tuned"
```

El soporte por clase y por split de cada run queda en `training_summary.json` bajo
`class_support`, y el F1 y el soporte por clase de cada cascada bajo `per_class`. Ninguna de las
dos claves filtra nada: una clase por debajo del umbral se **reporta** —evento
`class_support_below_floor`— y se conserva. Descartar Botnet subiría el F1-macro de 0.6467 a 0.7545
borrando la evidencia que lo justifica.

Requiere **~8 GiB de RAM**: `load_cicids2017` concatena los ocho CSV antes de muestrear, así que
`--sample-size` no gobierna el pico. `--class-sample-floor` tampoco: cambia **cómo** se reparte la
muestra, no cuándo se lee el dataset.

La composición de bloques de la §2 está medida sobre el **dataset completo** (sin `--sample-size`),
porque es un hecho sobre CICIDS2017 y no sobre una muestra concreta. La misma información sale de
cualquier run: el loader emite ahora el reparto por sub-etiqueta (`split_sublabel_distribution`)
además del reparto por clase, y es la traza que hace el problema auditable desde un log. Una línea
`split_class_distribution label=BruteForce train=6405 test=1373` parece perfectamente sana; las dos
líneas de sub-etiqueta que van debajo, con `FTP-Patator test=0`, son las que delatan que el corte
holdeó una herramienta entera.

---

## 9. Qué llevarse de aquí

1. **Un corte cronológico no es anti-fuga por serlo: lo es respecto a la unidad sobre la que
   corta.** Si la unidad agrega procesos que ocurrieron en serie, el corte se convierte en un
   holdout de esos procesos sin avisar.
2. **Un colapso de etiquetas es una decisión de reporte que no debe propagarse a la partición.**
   Aquí bastó conservar la etiqueta cruda como columna de procedencia para desacoplar las dos.
3. **Una caída grande de métrica al endurecer el protocolo es una hipótesis, no un diagnóstico.**
   La forma de convertirla en diagnóstico es construir el protocolo intermedio que aísla un factor
   — que fue lo que faltaba la primera vez.
4. **En una cascada, la contaminación del conjunto de validación de unas clases se propaga a
   todas** a través del umbral compartido de Stage 1. Las métricas por clase de una arquitectura de
   dos etapas no son independientes entre sí.
5. **El orden en que llegan los datos es una afirmación, y hay que comprobarla.** `order_idx` decía
   «orden de captura» y significaba «orden alfabético del nombre de fichero» (§11). Ningún test
   podía verlo, porque la propiedad que los tests comprobaban —monótono dentro de un fichero— era
   cierta. La comprobación que faltaba era una lista escrita a mano con el orden real.
6. **Un F1 por clase no se puede leer sin el recall de la etapa que la precede.** En una cascada, un
   0.00 puede significar «el clasificador no sabe nombrar la clase» o «el clasificador no vio ni una
   fila de la clase». Son diagnósticos opuestos y llevan a arreglos opuestos (§5).

---

## 10. Corolario (8 de agosto de 2026): el mismo error, un nivel más abajo

Arreglar la partición destapó un segundo error de la misma familia, y la relación entre los dos es
lo interesante.

`optimize_threshold` elige el umbral maximizando F-beta sobre validación y anotaba el punto de
operación **de ese mismo conjunto**. `evaluate()` copiaba esa anotación dentro del dict de métricas
de test. Las cifras `Stage 1 P / R / FPR` de las tablas de §3 y §4 eran, por tanto, de validación,
presentadas junto a Accuracy y F1-macro que sí eran de test.

Medido con el umbral congelado sobre ambos conjuntos:

| Protocolo | S1 recall (validación) | S1 recall (test) | Δ |
|---|---|---|---|
| `random` | 0.975 | 0.976 | **+0.001** |
| `sub-label` | 0.760 | **0.560** | **−0.200** |
| `class` | 0.770 | **0.573** | **−0.197** |

*(Cifras del código actual. En agosto, con el orden alfabético de ficheros, esta tabla era
0.978/0.979, 0.986/0.804 y 0.626/0.393. El hueco cambia de tamaño; el hallazgo —que solo existe
bajo partición temporal— no.)*

**La fila `random` es el experimento de control, y hace todo el trabajo de diagnóstico.** El sesgo
de selección —el umbral es el argmax de F-beta sobre los puntos de operación de ese conjunto— actúa
igual en los tres protocolos, porque la búsqueda es literalmente el mismo código. Si fuera la causa,
el hueco aparecería en las tres filas. Aparece solo en las temporales.

La causa es estructural y es *la misma que motivó este documento*: bajo el protocolo temporal
validación es el 15 % central de cada bloque y test el 15 % final, así que **validación está más
cerca de train en el tiempo que test**. El score del Isolation Forest deriva a lo largo de la ráfaga
de ataque. Con split aleatorio validación y test son intercambiables por construcción y sustituir
uno por otro es gratis; el protocolo temporal los vuelve dos momentos distintos, y ahí la
sustitución empieza a mentir.

Hay una ironía que conviene no perder: **el split temporal no creó este problema, lo hizo visible.**
Bajo el protocolo antiguo el atajo era correcto por accidente. Endurecer la partición invalidó una
suposición que nadie había escrito porque nunca había hecho falta.

Consecuencia práctica: el recall de Stage 1 es el techo duro de la cascada —lo que no marca sale
como BENIGN sin que Stage 2 lo vea— así que la fracción de ataques irrecuperables no era el 1,4 %
publicado sino el **19,6 %**. Factor 14, sobre la cifra de seguridad más cargada del diseño.

> Con el orden de captura corregido (§11) esa fracción es del **44,0 %** bajo el default, y del
> 18,1 % con el suelo de muestreo activado. La cifra empeora; el error que este corolario registra
> —medirla donde se elige el umbral— es el mismo, y es lo que permitió verla.

Ninguna métrica end-to-end cambia: `accuracy`, `f1_macro` y `f1_weighted` siempre salieron de
`predict(X_test)`. La descomposición de §4 se sostiene entera.

### El arreglo

| Fichero | Cambio |
|---|---|
| [`ml/flow/models/isolation_forest.py`](../../ml/flow/models/isolation_forest.py) | Nuevo `measure_operating_point(X, y, beta)`: mide el umbral **congelado** sobre un conjunto cualquiera. `_compute_operating_info` recalcula F-beta desde la matriz de confusión en vez de recibirla de la búsqueda, que es lo que permite reutilizarla fuera del conjunto de selección. El docstring de `get_operating_point_info` dice ahora que es un estadístico de ajuste |
| [`ml/flow/pipelines/cascade_xgb.py`](../../ml/flow/pipelines/cascade_xgb.py), [`cascade_mlp.py`](../../ml/flow/pipelines/cascade_mlp.py) | `evaluate()` mide Stage 1 sobre `X_test`. La búsqueda de validación solo se consulta para saber que hubo umbral y con qué beta |
| [`ml/training/train_cicids.py`](../../ml/training/train_cicids.py) | `stage1_operating_point` pasa a `{selection, test}`. Nada queda sin etiqueta |
| [`tests/unit/test_flow_models.py`](../../tests/unit/test_flow_models.py) | `TestMeasureOperatingPoint`: medir sobre el conjunto de selección reproduce el punto guardado; medir sobre otro cuenta *ese* otro; medir no mueve el umbral |
| [`tests/unit/test_train_cicids.py`](../../tests/unit/test_train_cicids.py) | El test que fija el hallazgo: `selection` suma `val_size`, `test` suma `test_size`, y `metrics.if_recall` es el de test |

### La lección, que es la de §9 desplazada un nivel

**Un número honesto sobre el conjunto equivocado sigue siendo un número deshonesto.** El primer
error puso la frontera de la partición en el sitio equivocado; este puso la frontera del *reporte*
en el sitio equivocado. Los dos pasaron una suite en verde durante semanas, porque ningún test puede
comprobar qué pregunta creías estar contestando. Lo único que los detectó, las dos veces, fue
construir el experimento de control que aísla un factor — §9.3 ya lo decía, y volvió a ser cierto.

---

## 11. Corolario 2 (22 de agosto de 2026): `order_idx` no era cronológico entre ficheros

Tercer error de la misma familia, encontrado al verificar los dos anteriores, y **el que más mueve
las cifras de este documento**.

`load_cicids2017` iteraba `sorted(data_path.glob("*.csv"))` y asignaba `order_idx` como un offset
acumulado sobre ese orden. Alfabéticamente:

```
Friday-…-Afternoon-DDos  <  Friday-…-Afternoon-PortScan  <  Friday-…-Morning
Thursday-…-Afternoon-Infilteration  <  Thursday-…-Morning-WebAttacks
```

Es decir, **las capturas de tarde recibían un `order_idx` menor que las de la mañana del mismo
día**. `order_idx` era monótono dentro de cada fichero —lo único que su docstring afirmaba— pero
entre ficheros codificaba el alfabeto, no el reloj.

### Por qué afecta a BENIGN y solo a BENIGN

Composición por fichero, contada sobre los CSV reales:

| Fichero (orden de captura) | Etiquetas crudas |
|---|---|
| `Monday-WorkingHours` | BENIGN 529.918 |
| `Tuesday-WorkingHours` | BENIGN 432.074 · FTP-Patator 7.938 · SSH-Patator 5.897 |
| `Wednesday-workingHours` | BENIGN 440.031 · DoS Hulk 231.073 · GoldenEye 10.293 · slowloris 5.796 · Slowhttptest 5.499 · Heartbleed 11 |
| `Thursday-Morning-WebAttacks` | BENIGN 168.186 · Brute Force 1.507 · XSS 652 · Sql Injection 21 |
| `Thursday-Afternoon-Infilteration` | BENIGN 288.566 · Infiltration 36 |
| `Friday-Morning` | BENIGN 189.067 · Bot 1.966 |
| `Friday-Afternoon-DDos` | BENIGN 97.718 · DDoS 128.027 |
| `Friday-Afternoon-PortScan` | BENIGN 127.537 · PortScan 158.930 |

**Ninguna etiqueta de ataque aparece en dos ficheros.** Los bloques de ataque son mono-fichero, así
que su corte cronológico era correcto pese al bug. Los bloques BENIGN de **jueves** (2 ficheros) y
**viernes** (3) no lo eran: el corte 70/15/15 los partía en un orden inventado. Y BENIGN es
exactamente lo que entrena Stage 1 y lo que compone su conjunto de validación.

### El experimento de control

El arreglo son tres líneas —`CAPTURE_FILE_ORDER` en `labels.py`, la clave de ordenación en
`loader.py`, y el orden resuelto emitido en `cicids_load_started` para que sea auditable desde un
log— pero mueve el titular, así que hay que atribuirlo en vez de afirmarlo. El control es el código
actual con `capture_file_rank` neutralizado, de forma que `sorted(glob, key=(rank, nombre))`
degenera en el orden alfabético anterior:

| Run | Accuracy | F1-macro | F1-weighted | S1 test P / R / FPR |
|---|---|---|---|---|
| Control, orden alfabético | 0.9612 | **0.7721** | 0.9597 | 0.250 / 0.804 / 0.491 |
| Publicado en agosto | 0.9612 | **0.7721** | 0.9597 | 0.250 / 0.804 / 0.491 |
| Código actual, orden de captura | 0.9194 | **0.6467** | 0.9052 | 0.335 / 0.560 / 0.226 |

El control reproduce las cifras de agosto **hasta el último decimal publicado**, lo que cierra dos
cosas a la vez: los runs de agosto eran reproducibles, y el movimiento del titular tiene una sola
causa. De paso confirma que el `labels=` de `evaluate()` —el primer arreglo de esta tanda— no mueve
nada aquí: las siete clases tienen soporte en test en todos los runs, así que promediar sobre la
lista canónica y promediar sobre `y_test ∪ y_pred` da el mismo número. Se arregló porque el
instrumento estaba mal, no porque la lectura lo estuviera.

### Qué significa que el titular baje

**0.7721 no era un resultado, era en parte un artefacto de ordenación.** El umbral del Isolation
Forest pasa de 0.0911 a 0.0655 y con él el recall de Stage 1 sobre test de 0.804 a 0.560. La
partición temporal se volvió más honesta y el número empeoró; es la tercera vez que eso pasa en
este documento, y sigue siendo la dirección correcta.

Un matiz que conviene no ocultar: el arreglo tiene **dos canales**, no uno. Cambiar el orden de
lectura cambia (a) cómo se cortan los bloques BENIGN multi-fichero y (b) qué filas concretas saca
`groupby().sample()`, que opera sobre un frame ordenado de otra forma. La fila `random` acota el
segundo: bajo split aleatorio no hay corte temporal en absoluto, y aun así el titular se mueve
+0.0256 (0.9197 → 0.9453), concentrado además en Botnet, una clase con 23 filas de test. El canal
que importa es el primero, pero el segundo no es cero y este documento no lo separa más fino.

---

## 12. El CV del tuning: el mismo defecto, dos niveles

`tune_hyperparameters` cortaba el train de Stage 2 con `StratifiedKFold(shuffle=True)`. Eso pone
casi-duplicados de una misma ráfaga a los dos lados de la frontera de cada fold — la fuga que la
partición existe para eliminar, reintroducida dentro del train. **No contaminaba ninguna métrica
publicada** (todas salen de `predict(X_test)` sobre el test temporal, y todos los runs de
referencia usaban `--skip-tuning`), pero sesgaba qué hiperparámetros ganaban la búsqueda: el tuning
respondía a una pregunta distinta de la que el run reporta.

`ml/flow/cv.py` lo sustituye por **folds de ventana expansiva por bloque**: con `cv_folds=K` cada
bloque `(capture_day, raw_label)` se corta en `K+1` trozos y el fold *k* entrena en los trozos
`0..k-1` y valida en el *k*. No es un K-fold sobre bloques —eso pondría trozos posteriores en el
train del fold, que es el mismo pecado con otro disfraz— y sale **más barato**: (1+2+3)/4 = 1.5n de
ajuste contra 2.0n del `StratifiedKFold(3)` que reemplaza. El precio a documentar es que los folds
entrenan sobre 1/4, 2/4 y 3/4 de las filas: el sesgo conservador de `TimeSeriesSplit`, que favorece
modelos más simples.

Medido sobre las anomalías reales que Stage 1 enruta (`--tune`, `sub-label`, sin suelo):

```
xgb_tuning_started        cv_folds=3 n_iter=20 n_samples=67789 temporal_folds=True
xgb_tuning_temporal_folds n_folds=3 n_classes=7
                          train_sizes=[16943, 33893, 50836]  train_classes=[7, 7, 7]
                          val_sizes=[16950, 16943, 16953]    val_classes=[7, 7, 7]
xgb_tuning_complete       best_params={'subsample': 1.0, 'n_estimators': 300, 'max_depth': 6,
                          'learning_rate': 0.2, 'colsample_bytree': 0.8}  f1_macro=0.831591
```

La ventana expande como debe (16,9k → 33,9k → 50,8k filas de train), las siete clases están en las
dos mitades de los tres folds, no se descartó ningún fold y no hubo fallback. Efecto sobre el test:
**0.6467 → 0.6584** de F1-macro. Reloj de pared del run completo: **602 s**, contra los 10.800 del
`activeDeadlineSeconds` del CronJob — el comentario del manifiesto que decía que la búsqueda no
cabía en el plazo estaba equivocado, y por partida doble: la entrada no son las 140k filas de la
muestra sino las 67.789 anomalías enrutadas.

### El scorer, que es la parte que se lee sola

`scoring="f1_macro"` promedia sobre `union(y_true, y_pred)` **de ese fold**. Con `StratifiedKFold`
era inocuo, porque todos los folds contienen todas las clases por construcción. Con folds
cronológicos, que una clase falte en un trozo de validación pasa de imposible a rutinario, y
entonces el conjunto de clases promediado varía entre folds y entre candidatos: los scores de la
búsqueda dejan de ser comparables entre sí. Se sustituye por
`make_scorer(f1_score, average="macro", labels=np.arange(n_classes), zero_division=0)`.

**Es literalmente el mismo defecto que el `f1_macro` titular sin `labels=`**, un nivel más abajo.
Arriba, `evaluate()` promediaba sobre las clases que aparecían en el test, y una clase sin soporte
desaparecía del macro en vez de puntuar 0. Abajo, el scorer promediaba sobre las clases que
aparecían en el fold. Los dos son la misma frase de sklearn —el default de `f1_score` es
`labels=None`— y los dos se arreglan pasando la lista canónica. Que vivieran en dos módulos
escritos con meses de diferencia es lo que hizo que se vieran como dos bugs y no como uno: **el
conjunto sobre el que se promedia un macro es una decisión, y omitirla es tomarla mal.**
