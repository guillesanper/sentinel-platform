"""Genera la presentación de defensa del proyecto CICIDS2017.

Construye `Presentacion_CICIDS2017.pptx` en la raíz del proyecto a partir de las
cifras de `memoria.md` / `output/results.txt` y de las figuras ya generadas en
`output/`. La presentación está dimensionada para una defensa de 8 minutos y se
ajusta a los cinco ejes exigidos en las instrucciones: problema, datos, técnicas,
resultados y análisis crítico (sin demos en directo ni teoría general).

Uso:
    python generar_presentacion.py

Requisitos:
    pip install python-pptx
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

# --------------------------------------------------------------------------- #
# Rutas y constantes
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
PPTX_PATH = ROOT / "Presentacion_CICIDS2017.pptx"

# Figuras reutilizadas (rutas relativas a output/).
FIG = {
    "class_distribution": OUTPUT / "eda" / "class_distribution.png",
    "feature_correlation": OUTPUT / "eda" / "feature_correlation.png",
    "feature_distributions": OUTPUT / "eda" / "feature_distributions.png",
    "cm_a": OUTPUT / "pipeline_a" / "confusion_matrix.png",
    "feat_importance": OUTPUT / "pipeline_a" / "feature_importance.png",
    "threshold_a": OUTPUT / "pipeline_a" / "threshold_curve.png",
    "cm_b": OUTPUT / "pipeline_b" / "confusion_matrix.png",
    "training_curves": OUTPUT / "pipeline_b" / "training_curves.png",
    "f1_per_class": OUTPUT / "comparison" / "f1_per_class.png",
    "metrics_bar": OUTPUT / "comparison" / "metrics_bar.png",
    "roc_curves": OUTPUT / "comparison" / "roc_curves.png",
    "pr_curves": OUTPUT / "comparison" / "pr_curves.png",
}

# Paleta sobria.
AZUL = RGBColor(0x1F, 0x3A, 0x5F)      # azul oscuro (títulos / acento)
AZUL_CLARO = RGBColor(0x2E, 0x6D, 0xB4)  # azul medio (subtítulos / cabecera tabla)
GRIS = RGBColor(0x33, 0x33, 0x33)      # texto cuerpo
GRIS_CLARO = RGBColor(0xEC, 0xEF, 0xF3)  # fondo de filas alternas
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)
VERDE = RGBColor(0x1E, 0x7A, 0x3C)     # resaltar ganador
ROJO = RGBColor(0xB3, 0x2C, 0x2C)      # resaltar problema

# Dimensiones de diapositiva 16:9.
SLIDE_W = Emu(12192000)
SLIDE_H = Emu(6858000)

FONT = "Calibri"


# --------------------------------------------------------------------------- #
# Helpers de construcción
# --------------------------------------------------------------------------- #
def _set_background(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_accent_bar(slide) -> None:
    """Barra de color fina en la parte superior, para identidad visual."""
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE.RECTANGLE
        Emu(0), Emu(0), SLIDE_W, Pt(10),
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = AZUL
    bar.line.fill.background()
    bar.shadow.inherit = False


def _add_title(slide, titulo: str) -> None:
    box = slide.shapes.add_textbox(Emu(685800), Emu(228600), Emu(10820400), Emu(800100))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = titulo
    run.font.size = Pt(30)
    run.font.bold = True
    run.font.name = FONT
    run.font.color.rgb = AZUL


def _set_notes(slide, notas: str) -> None:
    slide.notes_slide.notes_text_frame.text = notas


def _add_bullets(slide, bullets, left, top, width, height, size=18):
    """Añade una caja de texto con viñetas. `bullets` es lista de (texto, nivel)
    o de strings (nivel 0). Soporta resaltado con prefijos especiales:
    el texto que empieza por '**' y termina por '**' se pone en negrita."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in bullets:
        if isinstance(item, tuple):
            texto, nivel = item
        else:
            texto, nivel = item, 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = nivel
        p.space_after = Pt(6)
        # Detectar color especial mediante prefijos.
        color = GRIS
        if texto.startswith("[+]"):
            texto, color = texto[3:].strip(), VERDE
        elif texto.startswith("[-]"):
            texto, color = texto[3:].strip(), ROJO

        bold = nivel == 0
        prefix = "• " if nivel == 0 else "– "
        run = p.add_run()
        run.text = prefix + texto
        run.font.size = Pt(size if nivel == 0 else size - 2)
        run.font.name = FONT
        run.font.bold = bold and color is GRIS
        run.font.color.rgb = color
    return box


def add_content_slide(prs, titulo, bullets, notas, size=18):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _set_background(slide, BLANCO)
    _add_accent_bar(slide)
    _add_title(slide, titulo)
    _add_bullets(slide, bullets, Emu(685800), Emu(1143000),
                 Emu(10820400), Emu(5181600), size=size)
    _set_notes(slide, notas)
    return slide


def add_picture_slide(prs, titulo, img_key, bullets, notas, size=16):
    """Diapositiva con figura a la derecha y viñetas a la izquierda."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, BLANCO)
    _add_accent_bar(slide)
    _add_title(slide, titulo)
    if bullets:
        _add_bullets(slide, bullets, Emu(457200), Emu(1219200),
                     Emu(5486400), Emu(5029200), size=size)
        img_left, img_width = Emu(6172200), Emu(5715000)
    else:
        img_left, img_width = Emu(2238375), Emu(7715250)

    img = FIG[img_key]
    pic = slide.shapes.add_picture(str(img), img_left, Emu(1295400), width=img_width)
    # Centrar verticalmente si sobra espacio.
    max_h = Emu(4953000)
    if pic.height > max_h:
        ratio = max_h / pic.height
        pic.height = int(max_h)
        pic.width = int(pic.width * ratio)
        pic.left = int(img_left + (img_width - pic.width) / 2)
    _set_notes(slide, notas)
    return slide


def add_table_slide(prs, titulo, headers, rows, notas, bullets=None,
                    col_widths=None, highlight_winner_col=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, BLANCO)
    _add_accent_bar(slide)
    _add_title(slide, titulo)

    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_left = Emu(685800)
    tbl_top = Emu(1219200)
    tbl_width = Emu(10820400)
    tbl_height = Emu(457200 + 365760 * len(rows))
    gtbl = slide.shapes.add_table(n_rows, n_cols, tbl_left, tbl_top,
                                  tbl_width, tbl_height).table

    if col_widths:
        total = sum(col_widths)
        for j, w in enumerate(col_widths):
            gtbl.columns[j].width = Emu(int(int(tbl_width) * w / total))

    # Cabecera.
    for j, h in enumerate(headers):
        cell = gtbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = AZUL_CLARO
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = h
        run.font.bold = True
        run.font.size = Pt(16)
        run.font.name = FONT
        run.font.color.rgb = BLANCO

    # Filas.
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = gtbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = GRIS_CLARO if i % 2 == 0 else BLANCO
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            run = para.add_run()
            run.text = str(val)
            run.font.size = Pt(15)
            run.font.name = FONT
            run.font.color.rgb = GRIS
            if j == 0:
                run.font.bold = True
            if highlight_winner_col is not None and j == highlight_winner_col:
                run.font.bold = True
                run.font.color.rgb = VERDE

    if bullets:
        top = int(tbl_top) + int(tbl_height) + 228600
        _add_bullets(slide, bullets, tbl_left, Emu(top),
                     tbl_width, Emu(1500000), size=16)
    _set_notes(slide, notas)
    return slide


def add_title_slide(prs, titulo, subtitulo, meta, notas):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, AZUL)
    # Banda inferior de acento.
    band = slide.shapes.add_shape(1, Emu(0), Emu(5715000), SLIDE_W, Pt(14))
    band.fill.solid()
    band.fill.fore_color.rgb = AZUL_CLARO
    band.line.fill.background()
    band.shadow.inherit = False

    box = slide.shapes.add_textbox(Emu(685800), Emu(2057400), Emu(10820400), Emu(2400000))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = titulo
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = BLANCO

    p2 = tf.add_paragraph()
    p2.space_before = Pt(12)
    r2 = p2.add_run()
    r2.text = subtitulo
    r2.font.size = Pt(22)
    r2.font.name = FONT
    r2.font.color.rgb = RGBColor(0xC9, 0xDA, 0xEF)

    p3 = tf.add_paragraph()
    p3.space_before = Pt(24)
    r3 = p3.add_run()
    r3.text = meta
    r3.font.size = Pt(16)
    r3.font.name = FONT
    r3.font.color.rgb = RGBColor(0xA9, 0xC0, 0xDC)

    _set_notes(slide, notas)
    return slide


def add_section_slide(prs, titulo, subtitulo=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, AZUL)
    box = slide.shapes.add_textbox(Emu(685800), Emu(2743200), Emu(10820400), Emu(1500000))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = titulo
    r.font.size = Pt(36)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = BLANCO
    if subtitulo:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(12)
        r2 = p2.add_run()
        r2.text = subtitulo
        r2.font.size = Pt(18)
        r2.font.name = FONT
        r2.font.color.rgb = RGBColor(0xC9, 0xDA, 0xEF)
    return slide


# --------------------------------------------------------------------------- #
# Construcción de la presentación
# --------------------------------------------------------------------------- #
def check_figures() -> None:
    faltan = [str(p) for p in FIG.values() if not p.exists()]
    if faltan:
        print("ERROR: faltan figuras necesarias:", file=sys.stderr)
        for f in faltan:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)


def build() -> None:
    check_figures()
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # --- 1. Portada -------------------------------------------------------- #
    add_title_slide(
        prs,
        "Detección de Intrusiones de Red con Pipelines en Cascada",
        "Comparación experimental de dos arquitecturas sobre CICIDS2017",
        "Aprendizaje Automático · Junio 2026",
        notas=(
            "[~15 s] Presento mi proyecto: un sistema de detección de intrusiones "
            "de red sobre el dataset CICIDS2017, en el que comparo dos pipelines en "
            "cascada que se diferencian únicamente en el clasificador de segunda etapa."
        ),
    )

    # --- 2. El problema ---------------------------------------------------- #
    add_content_slide(
        prs,
        "El problema: detección de intrusiones (IDS)",
        [
            "Objetivo: clasificar cada flujo de red en BENIGN o en una de 6 familias de ataque",
            ("DoS, DDoS, Scan, BruteForce, WebAttack, Botnet", 1),
            "Los IDS basados en ML superan a las firmas estáticas ante variantes nuevas",
            "[-] Desafío central: desbalanceo de clases EXTREMO",
            ("BENIGN ≈ 83 % del tráfico vs. Botnet / WebAttack ≈ 0,08 %  →  ratio > 800:1", 1),
            ("La accuracy global es engañosa: un trivial 'todo BENIGN' lograría 83 %", 1),
            "Heterogeneidad de los ataques: no hay un único conjunto de features discriminante",
        ],
        notas=(
            "[~50 s] El problema es de clasificación supervisada multi-clase sobre tráfico "
            "de red. La dificultad no es el algoritmo, sino el desbalanceo: BENIGN es el 83 % "
            "y clases como Botnet o WebAttack apenas el 0,08 %, más de 800 a 1. Por eso la "
            "accuracy global no sirve como métrica y todo el diseño gira en torno a compensar "
            "ese desbalanceo. Además los ataques son heterogéneos: un DoS satura, un escaneo "
            "hace conexiones cortas, etc., así que no hay una sola feature que los separe a todos."
        ),
        size=18,
    )

    # --- 3. Objetivo y diseño en cascada ----------------------------------- #
    add_content_slide(
        prs,
        "Objetivo: comparar dos pipelines en cascada",
        [
            "Arquitectura de 2 etapas con un Stage 1 COMPARTIDO; solo cambia el Stage 2",
            "Stage 1 — Isolation Forest: filtro de anomalías (¿normal o anómalo?)",
            ("score ≥ umbral → BENIGN directo  |  score < umbral → al Stage 2", 1),
            "Stage 2 — clasificador multi-clase (¿qué tipo de ataque?):",
            ("[+] Pipeline A:  Isolation Forest + XGBoost", 1),
            ("[+] Pipeline B:  Isolation Forest + Red Neuronal implementada en NumPy puro", 1),
            "Comparación controlada del trade-off rendimiento vs. complejidad del Stage 2",
            "El Stage 2 incluye la clase BENIGN → puede corregir falsos positivos del Stage 1",
        ],
        notas=(
            "[~50 s] La idea central es la cascada de dos etapas. El Stage 1 es un detector de "
            "anomalías que actúa como filtro grueso: descarta tráfico benigno y solo enruta lo "
            "sospechoso al Stage 2. Ese Stage 1, un Isolation Forest, es idéntico en ambos "
            "pipelines. Lo único que cambia es el clasificador multi-clase: XGBoost en el "
            "Pipeline A, y una red neuronal que implementé desde cero en NumPy en el Pipeline B. "
            "Así la comparación es justa: misma entrada al Stage 2, distinto modelo. Importante: "
            "el Stage 2 puede devolver muestras a BENIGN, corrigiendo falsos positivos del filtro."
        ),
        size=18,
    )

    # --- 4. Los datos ------------------------------------------------------ #
    add_picture_slide(
        prs,
        "Los datos: CICIDS2017",
        "class_distribution",
        [
            "Benchmark estándar de IDS (Univ. New Brunswick, 2017)",
            "8 CSV · 5 días laborables · 2.830.743 flujos",
            "78 features numéricas de CICFlowMeter",
            ("duración, longitudes, IAT, flags TCP, tasas…", 1),
            "15 etiquetas originales → 7 clases canónicas",
            ("agrupación por familia (p. ej. 5 variantes de DoS → DoS)", 1),
            "[-] Desbalanceo visible: BENIGN domina; Botnet y WebAttack casi invisibles",
        ],
        notas=(
            "[~55 s] Uso CICIDS2017, el benchmark de referencia: tráfico real capturado durante "
            "cinco días en una red corporativa simulada, 2,83 millones de flujos y 78 features "
            "estadísticas calculadas por CICFlowMeter. Las 15 etiquetas originales las agrupo en "
            "7 clases canónicas por familia de ataque, porque etiquetas como Heartbleed tienen "
            "11 muestras y no permiten particiones fiables; agrupándolas el modelo aprende el "
            "concepto de ataque, no la herramienta. El gráfico confirma el desbalanceo: BENIGN "
            "domina y las clases minoritarias son invisibles a escala."
        ),
    )

    # --- 5. Preprocesado --------------------------------------------------- #
    add_table_slide(
        prs,
        "Preprocesado y limpieza (utils/loader.py)",
        ["Paso", "Detalle", "Filas"],
        [
            ["1. Carga latin-1", "corrige en-dash de 'Web Attack – XSS'", "2.830.743"],
            ["2. NaN / Inf", "divisiones por cero en tasas de flujo", "−2.867"],
            ["3. Duplicados", "antes del split → evita data leakage", "−307.078"],
            ["4. Filtrado clases", "umbrales mín. (todas las 7 pasan)", "—"],
            ["5. float32", "593 MB → 297 MB de RAM", "—"],
            ["6. Muestreo estratif.", "preserva la distribución por clase", "1.000.001"],
            ["7. Split 70/15/15", "estratificado doble", "700k/150k/150k"],
        ],
        notas=(
            "[~45 s] El preprocesado son siete pasos en orden estricto. Cargo en latin-1 para no "
            "corromper el guion de las etiquetas web. Elimino los Inf/NaN, que son artefactos de "
            "dividir por duración cero, y los duplicados: 307.000 filas, casi el 11 %. Esto es "
            "clave hacerlo ANTES del split, o la misma fila aparecería en train y test inflando "
            "las métricas. Convierto a float32 para halvar la memoria, tomo una muestra "
            "estratificada de un millón de filas y hago un split estratificado 70/15/15. La "
            "estratificación garantiza que las clases minoritarias estén en los tres conjuntos."
        ),
        col_widths=[3, 6, 2.5],
    )

    # --- 6. Stage 1: Isolation Forest -------------------------------------- #
    add_content_slide(
        prs,
        "Técnicas — Stage 1: Isolation Forest",
        [
            "Detector de anomalías no supervisado (200 árboles)",
            "Se entrena SOLO con muestras BENIGN → aprende el perfil de tráfico normal",
            ("no ve ataques → generaliza a ataques no vistos", 1),
            "Umbral óptimo por barrido de 300 candidatos, maximizando F-beta (β = 2)",
            ("β = 2 prioriza recall: perder un ataque es peor que una falsa alarma", 1),
            "Resultado del filtro (idéntico en A y B):",
            ("[+] Recall = 0,9682   ·   FNR = 0,0318", 1),
            ("[-] FPR = 0,6008   ·   Precisión = 0,2467", 1),
        ],
        notas=(
            "[~45 s] El Stage 1 es un Isolation Forest, entrenado únicamente con tráfico BENIGN, "
            "de modo que aprende qué es lo normal sin haber visto ataques. La decisión no usa el "
            "predict de sklearn sino un umbral propio, que optimizo sobre validación maximizando "
            "F-beta con beta 2, porque en un IDS el recall pesa más: un falso negativo no lo "
            "recupera nadie. El resultado: detecta el 97 % de los ataques, pero a costa de un FPR "
            "del 60 %. Esos falsos positivos los puede corregir el Stage 2; el problema "
            "irrecuperable es el 3,18 % de ataques que el filtro deja pasar como BENIGN."
        ),
        size=18,
    )

    # --- 7. Stage 2A: XGBoost ---------------------------------------------- #
    add_content_slide(
        prs,
        "Técnicas — Stage 2A: XGBoost (Pipeline A)",
        [
            "Gradient boosting de árboles (hasta 300, profundidad 6)",
            "Pesos de clase inversos a la frecuencia → compensa el desbalanceo",
            ("equivale a class_weight='balanced'", 1),
            "Tuning: RandomizedSearchCV (20 iteraciones × 3 folds estratificados)",
            ("criterio de selección: f1_macro sobre las clases de ataque", 1),
            "Early stopping (paciencia 20) con métrica mlogloss en validación",
            "Solo procesa las muestras anómalas que enruta el Stage 1",
        ],
        notas=(
            "[~35 s] En el Pipeline A el Stage 2 es XGBoost, gradient boosting de árboles. Para "
            "el desbalanceo asigno a cada muestra un peso inverso a la frecuencia de su clase. "
            "Hago búsqueda de hiperparámetros con RandomizedSearchCV, 20 iteraciones por 3 folds, "
            "optimizando f1-macro, y entreno el modelo final con early stopping. Solo ve las "
            "muestras que el Stage 1 marcó como anómalas."
        ),
        size=18,
    )

    # --- 8. Stage 2B: Red Neuronal NumPy ----------------------------------- #
    add_content_slide(
        prs,
        "Técnicas — Stage 2B: Red Neuronal en NumPy (Pipeline B)",
        [
            "MLP 78 → 256 → 128 → 64 → 7, implementado SIN frameworks de deep learning",
            "Todo el ciclo a mano: forward, backprop por regla de la cadena, Adam",
            "Regularización: Batch Normalization + Dropout invertido + weight decay L2",
            "Frente al desbalanceo:",
            ("Focal Loss (γ=2) + label smoothing + pesos de clase", 1),
            ("balanced batching + sobremuestreo de minoritarias con ruido", 1),
            "LR scheduling (warmup + ReduceLROnPlateau) y gradient clipping",
            "Valor formativo: obliga a entender los fundamentos matemáticos",
        ],
        notas=(
            "[~50 s] El Pipeline B reemplaza XGBoost por una red neuronal que implementé "
            "íntegramente en NumPy, sin PyTorch ni TensorFlow. Eso significa programar a mano el "
            "forward, la retropropagación con la regla de la cadena, el optimizador Adam, la "
            "Batch Normalization con estadísticas separadas para train e inferencia, el dropout "
            "invertido y la Focal Loss. La Focal Loss y el balanced batching atacan el "
            "desbalanceo, centrando el gradiente en los ejemplos difíciles. El objetivo no era "
            "solo competir con XGBoost, sino el valor formativo de entender las matemáticas por "
            "dentro."
        ),
        size=17,
    )

    # --- 9. Resultados globales -------------------------------------------- #
    add_table_slide(
        prs,
        "Resultados globales: Pipeline A vs. Pipeline B",
        ["Métrica", "Pipeline A (XGBoost)", "Pipeline B (NN NumPy)"],
        [
            ["Accuracy", "0,9936", "0,9685"],
            ["F1-macro (primaria)", "0,9516", "0,7497"],
            ["F1-weighted", "0,9936", "0,9736"],
            ["Tiempo (CPU)", "~85 min", "~39 min"],
        ],
        notas=(
            "[~50 s] Estos son los resultados sobre el test de 150.001 flujos. La métrica "
            "primaria es F1-macro, que da el mismo peso a cada clase y por tanto no se deja "
            "engañar por el desbalanceo. Pipeline A logra 0,95, superando el objetivo de 0,90. "
            "Pipeline B se queda en 0,75, por debajo del 0,80 de referencia. Fíjense en que la "
            "accuracy y la F1-weighted son muy parecidas entre ambos: ahí domina BENIGN. La "
            "diferencia real solo aparece en F1-macro, que es justamente donde se nota el "
            "rendimiento en las clases minoritarias."
        ),
        bullets=[
            "[+] Pipeline A supera el benchmark F1-macro > 0,90  ·  la brecha está en F1-macro (−0,20)",
        ],
        col_widths=[4, 4, 4],
        highlight_winner_col=1,
    )

    # --- 10. Resultados por clase ------------------------------------------ #
    add_picture_slide(
        prs,
        "Resultados por clase: ¿dónde está la diferencia?",
        "f1_per_class",
        [
            "Clases grandes: empate técnico",
            ("DDoS, DoS, BruteForce → F1 ≈ 0,96–0,99 en ambos", 1),
            "[-] El gap se concentra en las ultraminoritarias:",
            ("Botnet:  A=0,76  vs  B=0,16   (Δ −0,61)", 1),
            ("WebAttack:  A=0,96  vs  B=0,35   (Δ −0,61)", 1),
            ("Scan:  A=0,99  vs  B=0,85   (Δ −0,14)", 1),
            "→ No es un fallo global de B, sino en las clases con < 1.000 muestras",
        ],
        notas=(
            "[~50 s] Este gráfico es la clave del análisis. En las clases con miles de muestras "
            "—DDoS, DoS, BruteForce— los dos pipelines empatan, con F1 por encima de 0,96. Toda "
            "la diferencia de 20 puntos en F1-macro viene de las clases ultraminoritarias: en "
            "Botnet y WebAttack la red neuronal se desploma a 0,16 y 0,35, frente a 0,76 y 0,96 "
            "de XGBoost. Es decir, el problema de B no es global, está localizado en las clases "
            "con menos de mil muestras de entrenamiento."
        ),
    )

    # --- 11. Análisis crítico ---------------------------------------------- #
    add_content_slide(
        prs,
        "Análisis crítico: ¿por qué falla la red neuronal?",
        [
            "Patrón en B: recall altísimo pero PRECISIÓN colapsada en minoritarias",
            ("Botnet precisión 8 % · WebAttack 22 % → muchas falsas alarmas por acierto", 1),
            "Causa 1 — balanced batching + Focal Loss sobre poquísimas muestras:",
            ("ve las mismas pocas muestras repetidas → frontera de decisión laxa", 1),
            "Causa 2 — ruido del Stage 1: ~75.000 falsos positivos BENIGN que B debe filtrar",
            ("XGBoost gestiona ese ruido mejor con 300 árboles regularizados", 1),
            "Causa 3 — NumPy puro sin GPU ni framework optimizado",
            "[-] Límite del sistema: el FNR 3,18 % del Stage 1 fija el techo de recall (96,8 %)",
        ],
        notas=(
            "[~55 s] ¿Por qué falla B? El patrón es siempre el mismo: recall casi perfecto pero "
            "precisión hundida; en Botnet, 12 falsas alarmas por cada acierto. Tres causas. "
            "Primera: el balanced batching repite las pocas muestras minoritarias una y otra vez, "
            "y la red memoriza esas muestras sin aprender una frontera robusta. Segunda: el Stage "
            "1 le pasa unos 75.000 falsos positivos benignos como ruido, y XGBoost los filtra "
            "mejor con sus árboles regularizados. Tercera: NumPy puro no iguala décadas de "
            "optimización de XGBoost. Y una limitación que afecta a TODO el sistema: el 3,18 % de "
            "ataques que el Stage 1 deja pasar fija un techo de recall del 96,8 %, da igual lo "
            "bueno que sea el Stage 2."
        ),
        size=17,
    )

    # --- 12. Conclusiones -------------------------------------------------- #
    add_content_slide(
        prs,
        "Conclusiones y mejoras futuras",
        [
            "[+] Pipeline A (IF + XGBoost) es el sistema recomendado: F1-macro 0,95 (> 0,90)",
            "Pipeline B (0,75) es inoperante en Botnet/WebAttack, pero tiene valor formativo",
            ("implementar backprop, BatchNorm, Adam y Focal Loss a mano", 1),
            "Aprendizaje clave: en una cascada el cuello de botella es el Stage 1",
            ("optimizar el detector inicial importa más que mejorar el Stage 2", 1),
            "Mejoras futuras:",
            ("ensemble A+B · SMOTE/ADASYN · Stage 1 supervisado · selección de features", 1),
        ],
        notas=(
            "[~40 s] En conclusión, el sistema recomendado es el Pipeline A: F1-macro 0,95, "
            "supera el benchmark y discrimina bien todas las clases. El Pipeline B no llega, pero "
            "su valor es formativo: implementar la red desde cero obliga a entender la "
            "matemática. El aprendizaje principal es que en una cascada el cuello de botella "
            "irrecuperable es el Stage 1. Como líneas futuras: combinar ambos pipelines en un "
            "ensemble, usar SMOTE para las minoritarias, o sustituir el Isolation Forest por un "
            "clasificador binario supervisado para bajar el FPR. Gracias, quedo a sus preguntas."
        ),
        size=18,
    )

    # --- Diapositivas de respaldo ------------------------------------------ #
    add_section_slide(prs, "Diapositivas de respaldo", "Material de apoyo para preguntas del tribunal")

    add_picture_slide(prs, "Matriz de confusión — Pipeline A (XGBoost)", "cm_a", [],
                      notas="Backup: matriz de confusión 7×7 del Pipeline A sobre el test set.")
    add_picture_slide(prs, "Matriz de confusión — Pipeline B (NN NumPy)", "cm_b", [],
                      notas="Backup: matriz de confusión del Pipeline B. Se aprecia la sobre-predicción "
                            "de Scan y la confusión en Botnet/WebAttack.")
    add_picture_slide(prs, "Importancia de features — XGBoost", "feat_importance", [],
                      notas="Backup: top features por ganancia media. Lideran Subflow Bwd Bytes y "
                            "Subflow Fwd Packets, coherente con el EDA.")
    add_picture_slide(prs, "Curvas de entrenamiento — Red Neuronal", "training_curves", [],
                      notas="Backup: loss y accuracy por época. El val loss queda por debajo del train "
                            "loss por la Focal Loss + balanced batching. Mejor F1 en época 88.")
    add_picture_slide(prs, "Comparativa de métricas globales", "metrics_bar", [],
                      notas="Backup: barras de accuracy, F1-macro y F1-weighted de A vs B.")
    add_picture_slide(prs, "Curvas ROC (One-vs-Rest)", "roc_curves", [],
                      notas="Backup: curvas ROC por clase de ambos pipelines.")
    add_picture_slide(prs, "Curvas Precision-Recall", "pr_curves", [],
                      notas="Backup: curvas PR, más informativas que ROC con clases muy desbalanceadas.")
    add_picture_slide(prs, "EDA — Correlación entre features", "feature_correlation", [],
                      notas="Backup: mapa de calor de Pearson; bloques de multicolinealidad en IAT y "
                            "tiempos de flujo justifican Dropout/colsample.")
    add_picture_slide(prs, "EDA — Distribuciones por clase", "feature_distributions", [],
                      notas="Backup: KDE por clase de las features con mayor F de ANOVA.")
    add_picture_slide(prs, "Curva de umbral del Isolation Forest", "threshold_a", [],
                      notas="Backup: barrido de 300 umbrales, trade-off recall vs FPR; se elige el que "
                            "maximiza F-beta con beta 2.")

    prs.save(str(PPTX_PATH))
    n = len(prs.slides._sldIdLst)
    print(f"OK: presentacion generada con {n} diapositivas -> {PPTX_PATH}")


if __name__ == "__main__":
    build()
