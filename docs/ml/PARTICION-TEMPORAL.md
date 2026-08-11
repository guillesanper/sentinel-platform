# La partición temporal del track de flujos: qué medía y qué mide ahora

Este documento registra un error de concepto que llegó a estar en la rama, cómo se detectó y cómo
se corrigió. Se conserva porque la afirmación equivocada era **la afirmación estrella del track de
flujos**, y porque el mecanismo que la produjo es fácil de repetir en cualquier dataset que colapse
etiquetas antes de partir por tiempo.

Todas las cifras de este documento se midieron el **7 de agosto de 2026** sobre los CSV reales de
CICIDS2017 en `api/csvs/` (2.520.798 filas tras limpieza), con:

```powershell
python -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 `
  --pipeline xgb --skip-tuning --no-plots --no-mlflow --split <protocolo>
```

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
| `--split random` | 0.9944 | **0.9197** | 0.9945 | 0.251 / 0.979 / 0.593 |
| `--split temporal --temporal-block sub-label` *(nuevo default)* | 0.9612 | **0.7721** | 0.9597 | 0.250 / 0.804 / 0.491 |
| `--split temporal --temporal-block class` *(default anterior)* | 0.8807 | **0.2973** | 0.8483 | 0.679 / 0.393 / 0.038 |

> La columna de Stage 1 está medida sobre **test**. En la primera versión de este documento traía
> las cifras de **validación**, que es donde se *elige* el umbral; ver §10.

F1 por clase (`classification_report`, 2 decimales), con el número de herramientas que compone cada
clase y su soporte en test:

| Clase | Herram. | Soporte | `random` | `sub-label` | `class` |
|---|---|---|---|---|---|
| BENIGN | — | 24.936 | 1.00 | 0.98 | 0.94 |
| Botnet | 1 | 24 | 0.54 | **0.00** | **0.00** |
| BruteForce | 2 | 110 | 0.98 | 0.98 | **0.00** |
| DDoS | 1 | 1.524 | 0.99 | 0.98 | 0.63 |
| DoS | 5 | 2.308 | 0.98 | 0.84 | 0.44 |
| Scan | 1 | 1.080 | 0.99 | 0.80 | 0.01 |
| WebAttack | 3 | 26 | 0.96 | 0.83 | 0.06 |

---

## 4. La descomposición

```
F1-macro   random 0.9197  →  sub-label 0.7721  →  class 0.2973
           └──── 0.1476 ────┘└──────── 0.4748 ────────┘
              fuga (23,7 %)    artefacto taxonomía × tiempo (76,3 %)
```

**La fuga por casi-duplicados vale 0.148 de F1-macro, no 0.622.** El resto — el 76 % de la caída —
lo produce una decisión de taxonomía que se coló en la definición de la partición.

### Por qué el F1-macro del protocolo `class` no es un agregado coherente

Aunque midiera lo que se pretendía, el número seguiría sin poder promediarse. El protocolo `class`
solo exige generalizar a una herramienta nueva **en las clases que tienen más de una herramienta**:

* `BruteForce`, `DoS` y `WebAttack` responden a "¿generaliza a una herramienta no vista?".
* `Botnet`, `DDoS` y `Scan` tienen una sola herramienta, así que su bloque se parte **exactamente
  igual** bajo los dos protocolos y responden a "¿detecta lo que ya vio?".

Promediar las seis da un número que no responde a ninguna de las dos preguntas. Es la razón de
fondo por la que el protocolo se conserva pero **se reporta aparte, nunca como titular**.

### El artefacto no se queda en las clases multi-herramienta

Es el hallazgo que no esperábamos y el que más justifica el cambio de default. `Scan` y `DDoS`
tienen **una sola herramienta**: su partición es idéntica bit a bit bajo `sub-label` y bajo `class`.
Y sin embargo caen de 0.80 a 0.01 y de 0.98 a 0.63 al cambiar la clave de bloqueo.

La vía es el **punto de operación compartido de Stage 1**. El umbral del Isolation Forest se elige
maximizando F-beta sobre validación, y la composición de validación la fija la misma clave de
bloqueo. Bajo `class`, la búsqueda aterriza en un umbral radicalmente más conservador:

| Protocolo | Stage 1 recall (test) | Stage 1 FPR (test) |
|---|---|---|
| `sub-label` | 0.804 | 0.491 |
| `class` | **0.393** | 0.038 |

Con `class`, **el 61 % de los ataques no supera Stage 1 y por tanto nunca llega a Stage 2**.
Ningún acierto de Stage 2 puede recuperarlos, y las clases de una sola herramienta pagan la factura
igual que las demás. La cascada tiene un único umbral para todas las clases: contaminar la
validación de tres clases contamina el punto de operación de las siete.

---

## 5. Botnet: el caso donde el veredicto duro está entero

`Botnet` cae **0.54 → 0.00 en los dos protocolos temporales**.

Tiene una sola sub-etiqueta (`Bot`), un solo día y ~1.950 filas, así que aquí no hay confusión
posible con la novedad: su bloque se parte idénticamente bajo ambas claves y aun así el modelo pasa
de 0.54 a cero en cuanto los casi-duplicados dejan de cruzar la frontera train/test. **Ese 0.54 era
memorización pura.**

Es el único punto donde la lectura dura original se sostiene sin matices, y por eso se conserva
explícitamente. Con ~1.950 flujos en el dataset completo, la muestra de 200.000 le deja **108 filas
de entrenamiento y 24 de test**: no hay con qué aprender la clase una vez que no se puede
memorizar. El punto débil real del diseño está ahí —en las clases minoritarias concentradas en un
solo día— no en un F1-macro global de 0.28.

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

> **El cambio de default mueve el número titular de la rama: 0.2809 → 0.7721.** Es intencionado y
> es el objeto del cambio. La cifra que sustituye no es más favorable por casualidad: es la que
> corresponde a la pregunta que se creía estar haciendo desde el principio.

---

## 7. El arreglo, fichero por fichero

| Fichero | Cambio |
|---|---|
| [`ml/flow/loader.py`](../../ml/flow/loader.py) | Nueva columna de procedencia `raw_label`, poblada **antes** de que el mapeo canónico sobrescriba `Label`, y añadida a `META_COLUMNS` para que no pueda llegar al modelo. `_resolve_block_columns` elige la clave del bloque; `_temporal_cut` la recibe como parámetro. `clean_dataset` deduplica incluyendo `raw_label`. `_log_split_distribution` emite además el reparto por sub-etiqueta |
| [`ml/flow/config.py`](../../ml/flow/config.py) | `TEMPORAL_BLOCKS = ("sub-label", "class")` y el campo `temporal_block`, con default `"sub-label"` y validación en `__post_init__` |
| [`ml/flow/labels.py`](../../ml/flow/labels.py) | `normalize_raw_label()` — la reparación de mojibake **sin** el colapso. `Infiltration` deja de ser `Scan` (ver abajo) |
| [`ml/training/train_cicids.py`](../../ml/training/train_cicids.py) | Flag `--temporal-block`; el bloque `split` del `training_summary.json` registra la clave usada (`null` bajo `--split random`, que no tiene bloques) |
| [`tests/unit/test_flow_loader.py`](../../tests/unit/test_flow_loader.py) | `TestTemporalBlockKey` fija el hallazgo: con `sub-label` cada herramienta está en train, val y test; con `class` el test de una clase multi-herramienta se compone de una herramienta ausente de train |

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
# Los tres protocolos, ~35 s de entrenamiento cada uno (más ~1 min de carga de
# CSV) con los datos reales en api/csvs/
$py = "python"
& $py -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 --pipeline xgb `
    --skip-tuning --no-plots --no-mlflow --split random         --output-dir "$env:TEMP\flow-random"
& $py -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 --pipeline xgb `
    --skip-tuning --no-plots --no-mlflow --temporal-block class --output-dir "$env:TEMP\flow-coarse"
& $py -m ml.training.train_cicids --data-dir api/csvs --sample-size 200000 --pipeline xgb `
    --skip-tuning --no-plots --no-mlflow                        --output-dir models/flow
```

Requiere **~8 GiB de RAM**: `load_cicids2017` concatena los ocho CSV antes de muestrear, así que
`--sample-size` no gobierna el pico.

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
| `random` | 0.978 | 0.979 | **+0.001** |
| `sub-label` | 0.986 | **0.804** | **−0.182** |
| `class` | 0.626 | **0.393** | **−0.233** |

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
