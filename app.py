"""
Clasificador de dígitos manuscritos — Random Forest (Streamlit App)
===================================================================

Aplicación interactiva de aprendizaje supervisado para el reconocimiento
de dígitos manuscritos (0-9) a partir de imágenes de 8x8 píxeles.

Dataset: `load_digits` de scikit-learn (1797 imágenes en escala de grises).
Modelo: RandomForestClassifier (ensamble de árboles de decisión sin redes neuronales).

Instrucciones de ejecución:
    pip install streamlit scikit-learn matplotlib numpy pandas
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# Configuración general de la página y estilos visuales
# ============================================================
st.set_page_config(
    page_title="Clasificador de Dígitos — PUCV Mecánica",
    page_icon="logo_pucv.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS modernos para una presentación pulida
st.markdown(
    """
    <style>
    /* Estilos generales de tipografía y espaciado */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.88rem;
        font-weight: 500;
        color: #64748b;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-error {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .card-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Funciones de procesamiento de datos y entrenamiento
# ============================================================
@st.cache_data
def cargar_dataset():
    """Carga el dataset de dígitos manuscritos de scikit-learn."""
    digits = load_digits()
    return digits.data, digits.target, digits.images


def submuestrear_por_clase(X_train, y_train, conteos_por_clase, random_state=42):
    """
    Construye un subconjunto con una cantidad específica de ejemplos por dígito.
    Permite simular desbalance de clases de forma controlada.
    """
    rng = np.random.RandomState(random_state)
    indices_finales = []
    avisos = []
    for clase in np.unique(y_train):
        indices_clase = np.where(y_train == clase)[0]
        n_deseado = conteos_por_clase.get(int(clase), len(indices_clase))
        if n_deseado > len(indices_clase):
            avisos.append(
                f"Se solicitaron {n_deseado} ejemplos del dígito {clase}, pero solo "
                f"hay {len(indices_clase)} disponibles en el conjunto de entrenamiento. "
                "Se utilizarán todos los disponibles."
            )
            n_deseado = len(indices_clase)
        elegidos = rng.choice(indices_clase, size=n_deseado, replace=False)
        indices_finales.append(elegidos)
    indices_finales = np.concatenate(indices_finales)
    rng.shuffle(indices_finales)
    return X_train[indices_finales], y_train[indices_finales], avisos


def cargar_datos(X, y, config):
    """Divide los datos en entrenamiento y prueba, aplicando los filtros de configuración."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )
    avisos = []
    if config.get("train_class_counts"):
        X_train, y_train, avisos = submuestrear_por_clase(
            X_train, y_train, config["train_class_counts"], config["random_state"]
        )
    elif config["train_samples"] is not None and config["train_samples"] < len(X_train):
        rng = np.random.RandomState(config["random_state"])
        idx = rng.choice(len(X_train), size=config["train_samples"], replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    return X_train, X_test, y_train, y_test, avisos


def entrenar_modelo(X_train, y_train, config):
    """Entrena un clasificador Random Forest según la configuración indicada."""
    modelo = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        random_state=config["random_state"],
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    return modelo


# ============================================================
# Funciones auxiliares de visualización
# ============================================================
def fig_ejemplos_predichos(X, y_real, y_pred, n=9, cols=3):
    """Genera una cuadrícula con ejemplos de predicciones, destacando aciertos y errores."""
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(2.4 * cols, 2.5 * filas), dpi=130)
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        ax = axes[i]
        ax.imshow(X[i].reshape(8, 8), cmap="gray_r", interpolation="nearest")
        real, pred = y_real[i], y_pred[i]
        es_correcto = pred == real
        color = "#15803d" if es_correcto else "#dc2626"
        estado = "✓ Acierto" if es_correcto else "✗ Error"

        ax.set_title(f"Real: {real} | Pred: {pred}\n({estado})", color=color, fontsize=8.5, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.8 if not es_correcto else 1.0)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def fig_casos_mal_clasificados(X_err, y_real, y_pred, p_pred, p_real, max_items=16, cols=4):
    """Visualiza los casos mal clasificados mostrando la imagen y las probabilidades asociadas."""
    n = min(len(y_real), max_items)
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(filas, cols, figsize=(2.6 * cols, 2.8 * filas), dpi=130)
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        ax = axes[i]
        ax.imshow(X_err[i].reshape(8, 8), cmap="gray_r", interpolation="nearest")
        real = y_real[i]
        pred = y_pred[i]
        prob_pred_val = p_pred[i] * 100
        prob_real_val = p_real[i] * 100

        ax.set_title(
            f"Real: {real} ➔ Pred: {pred}\nP({pred})={prob_pred_val:.0f}% | P({real})={prob_real_val:.0f}%",
            color="#b91c1c",
            fontsize=8.5,
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#ef4444")
            spine.set_linewidth(2.0)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig


# ============================================================
# Carga de datos base
# ============================================================
X, y, IMAGENES = cargar_dataset()

# Encabezado principal con logo institucional PUCV a la derecha
col_hdr_main, col_hdr_logo = st.columns([5.5, 1])
with col_hdr_main:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <div class="main-title" style="margin: 0; line-height: 1.15;">Clasificador de Dígitos Manuscritos</div>
            <div style="color: #475569; font-size: 0.95rem; margin-top: 6px;">
                <strong>Escuela de Ingeniería Mecánica — PUCV</strong> · Random Forest sobre <code>load_digits</code> (1,797 imágenes de 8×8 píxeles).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_hdr_logo:
    st.image("logo_pucv.png", width=95)


# ============================================================
# Barra lateral: Configuración y parámetros
# ============================================================
with st.sidebar:
    st.header("⚙️ Parámetros del Modelo")

    test_size = st.slider(
        "Proporción de evaluación (test)",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Fracción del total del dataset que se reservará exclusivamente para evaluar la capacidad de generalización del modelo con datos no vistos.",
    )
    max_train_disponible = int(len(X) * (1 - test_size))

    st.markdown("---")
    st.subheader("📁 Conjunto de Entrenamiento")
    modo_datos = st.radio(
        "Modo de selección de muestras",
        ["Utilizar todos los datos", "Limitar cantidad total", "Desbalance por dígito"],
        help="Seleccione el método para conformar el conjunto de entrenamiento y observar su impacto en el rendimiento.",
    )

    train_samples = None
    train_class_counts = None

    if modo_datos == "Limitar cantidad total":
        train_samples = st.slider(
            "Cantidad de imágenes de entrenamiento",
            min_value=10,
            max_value=max_train_disponible,
            value=min(500, max_train_disponible),
            step=10,
            help="Reduce el tamaño total del entrenamiento para simular escenarios con escasez de datos etiquetados.",
        )
    elif modo_datos == "Desbalance por dígito":
        st.caption("Cantidad deseada de ejemplos por dígito (máximo ≈140 a 180 según la partición):")
        train_class_counts = {}
        columnas = st.columns(2)
        for digito in range(10):
            with columnas[digito % 2]:
                train_class_counts[digito] = st.slider(
                    f"Dígito {digito}",
                    min_value=0,
                    max_value=180,
                    value=140,
                    step=5,
                    key=f"count_{digito}",
                    help=f"Cantidad de ejemplos de entrenamiento asignados al dígito {digito}.",
                )

    st.markdown("---")
    st.subheader("🌲 Hiperparámetros de Random Forest")
    n_estimators = st.slider(
        "Número de árboles (n_estimators)",
        5,
        300,
        100,
        5,
        help="Cantidad de árboles de decisión independientes que votan en el ensamble. Más árboles aumentan la estabilidad y reducen la varianza sin causar sobreajuste.",
    )
    usar_max_depth = st.checkbox(
        "Limitar profundidad máxima (max_depth)",
        value=False,
        help="Si está desactivado (None), los árboles crecen hasta hojas puras. Si se activa, se controla la profundidad máxima para regularizar.",
    )
    max_depth = (
        st.slider(
            "Profundidad máxima",
            1,
            30,
            10,
            1,
            help="Longitud máxima de ramas. Valores pequeños regularizan el modelo contra sobreajuste; valores demasiado bajos causan subajuste.",
        )
        if usar_max_depth
        else None
    )
    min_samples_split = st.slider(
        "Mínimo de muestras para división (min_samples_split)",
        2,
        20,
        2,
        1,
        help="Mínimo de ejemplos requeridos en un nodo interno para permitir una nueva bifurcación. Valores más altos evitan divisiones sobreajustadas a casos aislados.",
    )
    min_samples_leaf = st.slider(
        "Mínimo de muestras en hoja (min_samples_leaf)",
        1,
        20,
        1,
        1,
        help="Mínimo de ejemplos requeridos en cada hoja terminal. Valores mayores a 1 suavizan las fronteras de decisión y previenen memorización.",
    )
    max_features = st.selectbox(
        "Características por división (max_features)",
        ["sqrt", "log2", None],
        index=0,
        help="'sqrt': evalúa √64 = 8 píxeles aleatorios por nodo (estándar para clasificación). 'log2': evalúa 6 píxeles. 'None': evalúa los 64 píxeles.",
    )
    random_state = st.number_input(
        "Semilla aleatoria (random_state)",
        value=42,
        step=1,
        help="Semilla del generador pseudoaleatorio para asegurar que las particiones y el entrenamiento sean 100% reproducibles.",
    )

    with st.expander("ℹ️ Resumen Rápido de Hiperparámetros"):
        st.markdown(
            """
            - **n_estimators**: Cantidad de árboles en el bosque (más árboles = mayor estabilidad).
            - **max_depth**: Profundidad máxima (evita memorizar ruido si se limita).
            - **min_samples_split**: Mínimo de muestras requeridas para crear nuevas ramas.
            - **min_samples_leaf**: Mínimo de muestras que deben quedar en cada hoja.
            - **max_features**: Píxeles evaluados al azar por nodo (`sqrt` = 8 píxeles).
            - **random_state**: Semilla para reproducibilidad de resultados.
            """
        )

    st.markdown("---")
    entrenar_click = st.button("🚀 Entrenar y Evaluar Modelo", type="primary", width="stretch")

CONFIG = {
    "test_size": test_size,
    "train_samples": train_samples,
    "train_class_counts": train_class_counts,
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "max_features": max_features,
    "random_state": int(random_state),
}


# ============================================================
# Lógica de entrenamiento persistida en st.session_state
# ============================================================
if entrenar_click or "resultados" not in st.session_state:
    with st.spinner("Entrenando clasificador Random Forest..."):
        Xtr, Xte, ytr, yte, avisos = cargar_datos(X, y, CONFIG)
        modelo = entrenar_modelo(Xtr, ytr, CONFIG)
        y_pred = modelo.predict(Xte)
        y_proba = modelo.predict_proba(Xte)

        # Identificación de aciertos y errores
        indices_errores = np.where(y_pred != yte)[0]

        st.session_state["resultados"] = {
            "Xtr": Xtr,
            "Xte": Xte,
            "ytr": ytr,
            "yte": yte,
            "avisos": avisos,
            "modelo": modelo,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "indices_errores": indices_errores,
            "config": dict(CONFIG),
        }

res = st.session_state["resultados"]
acc = accuracy_score(res["yte"], res["y_pred"])
total_test = len(res["yte"])
total_errores = len(res["indices_errores"])
total_aciertos = total_test - total_errores
tasa_error = (total_errores / total_test) if total_test > 0 else 0.0


# ============================================================
# Pestañas principales de la aplicación
# ============================================================
tab_datos, tab_como_opera, tab_rendimiento, tab_errores = st.tabs(
    [
        "📊 Exploración del Dataset",
        "🧠 ¿Cómo Opera el Modelo?",
        "🎯 Rendimiento del Modelo",
        f"❌ Casos Mal Clasificados ({total_errores})",
    ]
)


# ------------------------------------------------------------
# Pestaña 1: Exploración del Dataset
# ------------------------------------------------------------
with tab_datos:
    st.subheader("Visión General del Dataset")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(
            '<div class="metric-box"><div class="metric-val">1,797</div><div class="metric-lbl">Total de Imágenes</div></div>',
            unsafe_allow_html=True,
        )
    with col_kpi2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">{len(res["Xtr"])}</div><div class="metric-lbl">Muestras de Entrenamiento</div></div>',
            unsafe_allow_html=True,
        )
    with col_kpi3:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">{len(res["Xte"])}</div><div class="metric-lbl">Muestras de Prueba</div></div>',
            unsafe_allow_html=True,
        )
    with col_kpi4:
        st.markdown(
            '<div class="metric-box"><div class="metric-val">8 × 8</div><div class="metric-lbl">Resolución (64 Píxeles)</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_izq, col_der = st.columns([1.1, 1])

    with col_izq:
        st.subheader("Distribución de Clases (Entrenamiento)")
        X_tr_prev, X_te_prev, y_tr_prev, y_te_prev, avisos_prev = cargar_datos(X, y, CONFIG)
        for aviso in avisos_prev:
            st.warning(aviso)

        conteo_train = np.bincount(y_tr_prev, minlength=10)
        fig_dist, ax_dist = plt.subplots(figsize=(6.5, 3.8), dpi=130)
        barras = ax_dist.bar(range(10), conteo_train, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
        ax_dist.set_xlabel("Dígito", fontsize=10, fontweight="bold")
        ax_dist.set_ylabel("Cantidad de muestras", fontsize=10, fontweight="bold")
        ax_dist.set_xticks(range(10))
        ax_dist.grid(axis="y", linestyle="--", alpha=0.3)
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["right"].set_visible(False)

        # Añadir valores numéricos sobre las barras
        for bar in barras:
            altura = bar.get_height()
            ax_dist.annotate(
                f"{int(altura)}",
                xy=(bar.get_x() + bar.get_width() / 2, altura),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        fig_dist.tight_layout()
        st.pyplot(fig_dist)

    with col_der:
        st.subheader("Dígito Promedio (Firma de Intensidad)")
        st.caption("Promedio de píxeles para cada dígito en todo el dataset:")
        fig_prom, axes_prom = plt.subplots(2, 5, figsize=(6.5, 3.2), dpi=130)
        for dig in range(10):
            ax = axes_prom[dig // 5, dig % 5]
            promedio = IMAGENES[y == dig].mean(axis=0)
            ax.imshow(promedio, cmap="Blues", interpolation="nearest")
            ax.set_title(f"Dígito {dig}", fontsize=9, fontweight="bold")
            ax.axis("off")
        fig_prom.tight_layout()
        st.pyplot(fig_prom)

    st.markdown("---")
    st.subheader("Ejemplos Aleatorios del Dataset")
    rng_ej = np.random.RandomState(42)
    indices_muestra = rng_ej.choice(len(X), size=16, replace=False)
    fig_muestras, axes_m = plt.subplots(2, 8, figsize=(11, 3.0), dpi=130)
    for ax, i in zip(axes_m.ravel(), indices_muestra):
        ax.imshow(IMAGENES[i], cmap="gray_r", interpolation="nearest")
        ax.set_title(f"Dígito: {y[i]}", fontsize=9)
        ax.axis("off")
    fig_muestras.tight_layout()
    st.pyplot(fig_muestras)


# ------------------------------------------------------------
# Pestaña 2: ¿Cómo Opera el Modelo? (Información que toma y proceso)
# ------------------------------------------------------------
with tab_como_opera:
    st.subheader("💡 ¿Qué Información Toma el Modelo y Cómo la Procesa?")

    # 3 Tarjetas conceptuales
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.markdown(
            """
            <div class="card-container" style="height: 100%;">
                <h4 style="color: #2563eb; margin-top:0;">1. Entrada: 64 Píxeles</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                    Cada imagen tiene <strong>8×8 píxeles</strong> en escala de grises. 
                    El algoritmo no analiza imágenes bidimensionales como los humanos, sino que 
                    la matriz se <strong>aplana en un vector de 64 números continuos</strong>:
                </p>
                <div style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 0.82rem; color: #0f172a;">
                    x = [p₀, p₁, p₂, ..., p₆₃]
                </div>
                <p style="font-size: 0.84rem; color: #64748b; margin-top: 8px;">
                    Cada valor oscila entre <strong>0</strong> (fondo blanco) y <strong>16</strong> (trazo negro de máxima intensidad).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp_col2:
        st.markdown(
            """
            <div class="card-container" style="height: 100%;">
                <h4 style="color: #16a34a; margin-top:0;">2. Bosque de Árboles</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                    Random Forest entrena un conjunto de <strong>árboles de decisión independientes</strong>.
                    Cada árbol evalúa subconjuntos aleatorios de píxeles (por ejemplo, √64 = 8) y aprende reglas binarias:
                </p>
                <div style="background: #f0fdf4; border-left: 3px solid #16a34a; padding: 6px 10px; font-size: 0.84rem; color: #166534;">
                    <em>"¿El píxel central p₃₆ tiene intensidad > 8.5?"</em>
                </div>
                <p style="font-size: 0.84rem; color: #64748b; margin-top: 8px;">
                    Si es sí, descarta números sin trazo central (como el 0); si es no, evalúa otras ramas del árbol.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with exp_col3:
        st.markdown(
            """
            <div class="card-container" style="height: 100%;">
                <h4 style="color: #9333ea; margin-top:0;">3. Votación y Decisión</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                    Cada árbol del bosque emite un voto individual por la clase que considera más probable. 
                    La decisión final se toma por <strong>consenso mayoritario</strong> o promedio de probabilidades:
                </p>
                <div style="background: #faf5ff; border-left: 3px solid #9333ea; padding: 6px 10px; font-size: 0.84rem; color: #6b21a8;">
                    <em>Predicción = Dígito con mayor cantidad de votos</em>
                </div>
                <p style="font-size: 0.84rem; color: #64748b; margin-top: 8px;">
                    Combinar decenas o cientos de árboles neutraliza los errores individuales de cada uno.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("🔬 Demostración Interactiva Paso a Paso")
    st.caption("Seleccione una muestra del conjunto de prueba para observar la información exacta que procesa el modelo:")

    idx_demo = st.slider(
        "Muestra a examinar (índice en conjunto de prueba):",
        min_value=0,
        max_value=len(res["Xte"]) - 1,
        value=0,
        step=1,
    )

    muestra_img = res["Xte"][idx_demo].reshape(8, 8)
    muestra_vec = res["Xte"][idx_demo]
    real_demo = res["yte"][idx_demo]
    pred_demo = res["y_pred"][idx_demo]
    proba_demo = res["y_proba"][idx_demo]
    es_acierto = (pred_demo == real_demo)

    demo_col1, demo_col2, demo_col3 = st.columns([1, 1.1, 1.2])

    with demo_col1:
        st.markdown("**Paso 1: Imagen Original 2D (8×8)**")
        fig_demo1, ax_demo1 = plt.subplots(figsize=(3.4, 3.4), dpi=130)
        ax_demo1.imshow(muestra_img, cmap="gray_r", interpolation="nearest")
        color_borde = "#16a34a" if es_acierto else "#dc2626"
        for spine in ax_demo1.spines.values():
            spine.set_edgecolor(color_borde)
            spine.set_linewidth(2.5)
        ax_demo1.set_title(f"Etiqueta Real: {real_demo}\nPredicción: {pred_demo}", fontsize=10, fontweight="bold", color=color_borde)
        ax_demo1.set_xticks(range(8))
        ax_demo1.set_yticks(range(8))
        ax_demo1.set_xlabel("Columna (0-7)", fontsize=8)
        ax_demo1.set_ylabel("Fila (0-7)", fontsize=8)
        fig_demo1.tight_layout()
        st.pyplot(fig_demo1)

    with demo_col2:
        st.markdown("**Paso 2: Matriz Numérica de Píxeles (0 a 16)**")
        fig_demo2, ax_demo2 = plt.subplots(figsize=(3.6, 3.4), dpi=130)
        ax_demo2.imshow(muestra_img, cmap="Blues", interpolation="nearest")
        for f in range(8):
            for c in range(8):
                val = int(muestra_img[f, c])
                txt_color = "white" if val > 8 else "black"
                ax_demo2.text(c, f, str(val), ha="center", va="center", color=txt_color, fontsize=7.5, fontweight="bold")
        ax_demo2.set_title("Valores de Intensidad (0-16)", fontsize=9.5, fontweight="bold")
        ax_demo2.set_xticks([])
        ax_demo2.set_yticks([])
        fig_demo2.tight_layout()
        st.pyplot(fig_demo2)

    with demo_col3:
        st.markdown("**Paso 3: Distribución de Votos / Probabilidades**")
        fig_demo3, ax_demo3 = plt.subplots(figsize=(4.2, 3.4), dpi=130)
        colores_barras = ["#2563eb" if d == pred_demo else "#94a3b8" for d in range(10)]
        if not es_acierto:
            colores_barras[pred_demo] = "#dc2626"
            colores_barras[real_demo] = "#16a34a"
        barras_prob = ax_demo3.bar(range(10), proba_demo * 100, color=colores_barras, edgecolor="#1e293b", alpha=0.9)
        ax_demo3.set_xlabel("Dígito Candidato", fontsize=8.5, fontweight="bold")
        ax_demo3.set_ylabel("Probabilidad (%)", fontsize=8.5, fontweight="bold")
        ax_demo3.set_xticks(range(10))
        ax_demo3.set_ylim(0, 105)
        ax_demo3.grid(axis="y", linestyle="--", alpha=0.3)
        ax_demo3.spines["top"].set_visible(False)
        ax_demo3.spines["right"].set_visible(False)

        for bp in barras_prob:
            h = bp.get_height()
            if h > 3:
                ax_demo3.annotate(
                    f"{h:.0f}%",
                    xy=(bp.get_x() + bp.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )
        fig_demo3.tight_layout()
        st.pyplot(fig_demo3)

    st.markdown("**Paso 4: Vector Aplanado de 64 Características**")
    st.caption("Esta es la fila unidimensional exacta que ingresa a la función matemática `RandomForestClassifier.predict()`:")

    fig_vec, ax_vec = plt.subplots(figsize=(10, 1.3), dpi=130)
    ax_vec.imshow(muestra_vec.reshape(1, 64), cmap="Blues", aspect="auto")
    ax_vec.set_yticks([])
    ax_vec.set_xticks([0, 7, 15, 23, 31, 39, 47, 55, 63])
    ax_vec.set_xticklabels(["p₀", "p₇", "p₁₅", "p₂₃", "p₃₁", "p₃₉", "p₄₇", "p₅₅", "p₆₃"], fontsize=8)
    ax_vec.set_title("Vector de 64 dimensiones (1 muestra × 64 variables)", fontsize=9, fontweight="bold")
    fig_vec.tight_layout()
    st.pyplot(fig_vec)

    st.markdown("---")
    col_imp1, col_imp2 = st.columns([1, 1.2])
    with col_imp1:
        st.subheader("🎯 ¿Cuáles Píxeles Son Más Determinantes?")
        st.write(
            """
            Random Forest permite medir la **importancia de cada característica** (*Feature Importance*), 
            es decir, cuánto contribuye cada píxel a reducir la impureza y clasificar correctamente:
            - **Zona central e intermedia (colores cálidos / oscuros)**: Concentran la mayor variación entre dígitos (por ejemplo, el centro de un 8 vs. el hueco de un 0). Son los píxeles más decisivos para el modelo.
            - **Bordes exteriores (colores claros)**: Casi siempre están en 0 (fondo blanco), por lo que apenas aportan información para distinguir entre números.
            """
        )
    with col_imp2:
        importancias = res["modelo"].feature_importances_.reshape(8, 8)
        fig_imp, ax_imp = plt.subplots(figsize=(4.6, 3.4), dpi=130)
        im_heat = ax_imp.imshow(importancias, cmap="YlOrRd", interpolation="nearest")
        cbar = fig_imp.colorbar(im_heat, ax=ax_imp, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.set_label("Importancia Relativa", fontsize=8)
        ax_imp.set_title("Importancia de Píxeles en el Bosque", fontsize=9.5, fontweight="bold")
        ax_imp.set_xticks(range(8))
        ax_imp.set_yticks(range(8))
        ax_imp.set_xlabel("Columna (0-7)", fontsize=8)
        ax_imp.set_ylabel("Fila (0-7)", fontsize=8)
        fig_imp.tight_layout()
        st.pyplot(fig_imp)

    # Sección didáctica: Explicación de todos los hiperparámetros
    st.markdown("---")
    st.subheader("📚 Guía Completa de Hiperparámetros de Random Forest")
    st.markdown(
        "A continuación se explica la función conceptual y práctica de cada parámetro disponible en la barra lateral, "
        "así como su impacto en el equilibrio entre **sesgo y varianza** (*underfitting* vs. *overfitting*):"
    )

    hp_col1, hp_col2 = st.columns(2)

    with hp_col1:
        st.markdown(
            """
            <div class="card-container">
                <h4 style="color: #0284c7; margin-top: 0;">🌲 <code>n_estimators</code> (Número de árboles)</h4>
                <p style="font-size: 0.88rem; color: #334155; margin-bottom: 6px;">
                    <strong>¿Qué controla?</strong> La cantidad de árboles de decisión independientes que componen el bosque y votan por el dígito final.
                </p>
                <ul style="font-size: 0.85rem; color: #64748b; margin-top: 0; padding-left: 18px;">
                    <li><strong>Al aumentarlo (ej. 100 a 300):</strong> Reduce la varianza del modelo, suaviza las superficies de decisión y estabiliza las predicciones. En Random Forest, agregar más árboles <em>no produce sobreajuste</em>, pero sí incrementa el tiempo de cómputo.</li>
                    <li><strong>Al reducirlo (ej. 5 a 20):</strong> Entrenamiento casi instantáneo, pero con mayor variabilidad estadística en los resultados.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card-container">
                <h4 style="color: #0284c7; margin-top: 0;">📏 <code>max_depth</code> (Profundidad máxima)</h4>
                <p style="font-size: 0.88rem; color: #334155; margin-bottom: 6px;">
                    <strong>¿Qué controla?</strong> La cantidad máxima de preguntas sucesivas (niveles) que cada árbol puede formular desde la raíz hasta una hoja.
                </p>
                <ul style="font-size: 0.85rem; color: #64748b; margin-top: 0; padding-left: 18px;">
                    <li><strong>Sin límite (<code>None</code>):</strong> Los árboles crecen libremente hasta que todas las hojas sean puras. Puede memorizar ruido si las imágenes tienen anomalías.</li>
                    <li><strong>Limitado (ej. 5 a 10):</strong> Actúa como regularizador. Si es demasiado bajo (&lt; 4), causará <em>subajuste (underfitting)</em> al impedir que el árbol capture la complejidad de los trazos.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card-container">
                <h4 style="color: #0284c7; margin-top: 0;">📊 <code>test_size</code> (Proporción de evaluación)</h4>
                <p style="font-size: 0.88rem; color: #334155; margin-bottom: 6px;">
                    <strong>¿Qué controla?</strong> La fracción del dataset (ej. 20% = 360 imágenes) reservada exclusivamente para evaluar el modelo.
                </p>
                <ul style="font-size: 0.85rem; color: #64748b; margin-top: 0; padding-left: 18px;">
                    <li>Garantiza una estimación no sesgada del rendimiento real con datos nuevos que el modelo nunca vio durante el entrenamiento.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hp_col2:
        st.markdown(
            """
            <div class="card-container">
                <h4 style="color: #0284c7; margin-top: 0;">✂️ <code>min_samples_split</code> (Mínimo para dividir)</h4>
                <p style="font-size: 0.88rem; color: #334155; margin-bottom: 6px;">
                    <strong>¿Qué controla?</strong> El número mínimo de ejemplos de entrenamiento requeridos en un nodo interno para que se le permita crear dos nuevas ramas.
                </p>
                <ul style="font-size: 0.85rem; color: #64748b; margin-top: 0; padding-left: 18px;">
                    <li><strong>Valor bajo (2):</strong> Permite divisiones muy detalladas y árboles profundos.</li>
                    <li><strong>Valor alto (ej. 10 a 20):</strong> Impide que el árbol cree ramificaciones exclusivas para 1 o 2 imágenes raras, favoreciendo reglas más generales.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card-container">
                <h4 style="color: #0284c7; margin-top: 0;">🍃 <code>min_samples_leaf</code> (Mínimo por hoja)</h4>
                <p style="font-size: 0.88rem; color: #334155; margin-bottom: 6px;">
                    <strong>¿Qué controla?</strong> La cantidad mínima de muestras que deben residir en cada hoja terminal para considerarse válida.
                </p>
                <ul style="font-size: 0.85rem; color: #64748b; margin-top: 0; padding-left: 18px;">
                    <li><strong>Valor bajo (1):</strong> Cada hoja puede representar un único ejemplo; máxima sensibilidad a detalles específicos.</li>
                    <li><strong>Valor alto (ej. 4 a 10):</strong> Suaviza el modelo, obligando a que cada respuesta provenga de un grupo estadísticamente representativo de muestras.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card-container">
                <h4 style="color: #0284c7; margin-top: 0;">🔍 <code>max_features</code> (Características por división)</h4>
                <p style="font-size: 0.88rem; color: #334155; margin-bottom: 6px;">
                    <strong>¿Qué controla?</strong> El número de píxeles seleccionados al azar que cada árbol puede evaluar al buscar la mejor pregunta en cada nodo.
                </p>
                <ul style="font-size: 0.85rem; color: #64748b; margin-top: 0; padding-left: 18px;">
                    <li><strong><code>'sqrt'</code> (Recomendado):</strong> Evalúa √64 = 8 píxeles. Descorrelaciona los árboles entre sí, maximizando la sinergia y potencia del bosque.</li>
                    <li><strong><code>'log2'</code>:</strong> Evalúa log₂(64) = 6 píxeles. Mayor diversidad entre árboles individuales.</li>
                    <li><strong><code>None</code>:</strong> Evalúa los 64 píxeles en cada nodo. Los árboles son más homogéneos entre sí, reduciendo el beneficio del ensamble.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card-container" style="background: #f8fafc; border: 1px dashed #cbd5e1;">
            <h5 style="color: #334155; margin-top: 0;">🎲 <code>random_state</code> (Semilla pseudoaleatoria)</h5>
            <p style="font-size: 0.85rem; color: #64748b; margin-bottom: 0;">
                Fija la secuencia de números aleatorios para la división train/test y la selección de subconjuntos de datos y píxeles en cada árbol. 
                Garantiza la <strong>reproducibilidad científica</strong>: dos personas que usen el mismo <code>random_state</code> obtendrán exactamente el mismo modelo y las mismas métricas.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Pestaña 3: Rendimiento del Modelo
# ------------------------------------------------------------
with tab_rendimiento:
    for aviso in res["avisos"]:
        st.warning(aviso)

    # Tarjetas KPI superiores
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val" style="color: #2563eb;">{acc:.1%}</div>'
            '<div class="metric-lbl">Exactitud Global (Accuracy)</div></div>',
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val" style="color: #16a34a;">{total_aciertos}</div>'
            '<div class="metric-lbl">Predicciones Correctas</div></div>',
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val" style="color: {"#dc2626" if total_errores > 0 else "#16a34a"};">{total_errores}</div>'
            f'<div class="metric-lbl">Errores ({tasa_error:.1%})</div></div>',
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">{total_test}</div>'
            '<div class="metric-lbl">Total Evaluado (Prueba)</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_matriz, col_ejemplos = st.columns([1.05, 1])

    with col_matriz:
        st.subheader("Matriz de Confusión")
        st.caption("Fila: Dígito real · Columna: Dígito predicho por el modelo")
        cm = confusion_matrix(res["yte"], res["y_pred"])
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 5.0), dpi=130)
        disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title("Matriz de Confusión", fontsize=11, fontweight="bold", pad=10)
        ax_cm.set_xlabel("Predicción del Modelo", fontsize=10, fontweight="bold")
        ax_cm.set_ylabel("Etiqueta Real", fontsize=10, fontweight="bold")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

    with col_ejemplos:
        st.subheader("Muestra de Predicciones")
        st.caption("Verde = Acierto · Rojo = Error")
        rng_pred = np.random.RandomState(7)
        idx_eval_muestra = rng_pred.choice(len(res["Xte"]), size=min(9, len(res["Xte"])), replace=False)
        fig_pred = fig_ejemplos_predichos(
            res["Xte"][idx_eval_muestra],
            res["yte"][idx_eval_muestra],
            res["y_pred"][idx_eval_muestra],
            n=len(idx_eval_muestra),
            cols=3,
        )
        st.pyplot(fig_pred)

    st.markdown("---")
    st.subheader("Métricas Detalladas por Dígito")
    reporte_dict = classification_report(
        res["yte"], res["y_pred"], digits=3, output_dict=True, zero_division=0
    )
    df_rep = pd.DataFrame(reporte_dict).transpose()

    # Formateo amigable en español
    df_rep.rename(
        columns={
            "precision": "Precisión",
            "recall": "Exhaustividad (Recall)",
            "f1-score": "F1-Score",
            "support": "Muestras (Soporte)",
        },
        inplace=True,
    )
    df_rep.rename(
        index={
            "accuracy": "Exactitud global",
            "macro avg": "Promedio macro",
            "weighted avg": "Promedio ponderado",
        },
        inplace=True,
    )

    st.dataframe(
        df_rep.style.format(
            {
                "Precisión": "{:.3f}",
                "Exhaustividad (Recall)": "{:.3f}",
                "F1-Score": "{:.3f}",
                "Muestras (Soporte)": "{:.0f}",
            }
        ),
        width="stretch",
    )


# ------------------------------------------------------------
# Pestaña 3: Casos Mal Clasificados
# ------------------------------------------------------------
with tab_errores:
    st.subheader("Diagnóstico y Análisis de Casos Mal Clasificados")

    if total_errores == 0:
        st.success(
            "🎉 **¡Clasificación perfecta!** No se registraron errores en el conjunto de prueba con la configuración actual."
        )
    else:
        st.caption(
            "Visualice las muestras que generaron confusión en el modelo, junto con la probabilidad asignada a la predicción errónea y a la clase correcta."
        )

        # Preparación de datos de errores
        idx_err = res["indices_errores"]
        X_err = res["Xte"][idx_err]
        y_real_err = res["yte"][idx_err]
        y_pred_err = res["y_pred"][idx_err]
        probs_matrix = res["y_proba"][idx_err]

        # Probabilidad de la clase predicha y de la clase real
        clases_modelo = list(res["modelo"].classes_)
        p_pred_err = np.array([
            probs_matrix[i, clases_modelo.index(y_pred_err[i])]
            for i in range(len(idx_err))
        ])
        p_real_err = np.array([
            probs_matrix[i, clases_modelo.index(y_real_err[i])]
            if y_real_err[i] in clases_modelo else 0.0
            for i in range(len(idx_err))
        ])

        # Fila de métricas resumen de errores
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color: #dc2626;">{total_errores}</div>'
                f'<div class="metric-lbl">Total de Errores ({tasa_error:.1%})</div></div>',
                unsafe_allow_html=True,
            )

        # Dígito real con mayor número de errores
        conteo_err_por_digito = pd.Series(y_real_err).value_counts()
        digito_mas_dificil = conteo_err_por_digito.index[0]
        n_max_err = conteo_err_por_digito.iloc[0]

        with col_e2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color: #b91c1c;">Dígito {digito_mas_dificil}</div>'
                f'<div class="metric-lbl">Mayor Dificultad ({n_max_err} errores)</div></div>',
                unsafe_allow_html=True,
            )

        confianza_promedio_error = np.mean(p_pred_err) * 100
        with col_e3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color: #d97706;">{confianza_promedio_error:.1f}%</div>'
                '<div class="metric-lbl">Confianza Media en Predicciones Erróneas</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualización de distribución de errores por dígito real
        col_g1, col_g2 = st.columns([1.1, 1])

        with col_g1:
            st.subheader("Distribución de Errores por Dígito Real")
            errores_por_digito = [np.sum(y_real_err == d) for d in range(10)]
            fig_err_bar, ax_err_bar = plt.subplots(figsize=(6.5, 3.2), dpi=130)
            barras_err = ax_err_bar.bar(range(10), errores_por_digito, color="#ef4444", alpha=0.85, edgecolor="#b91c1c")
            ax_err_bar.set_xlabel("Dígito Real", fontsize=10, fontweight="bold")
            ax_err_bar.set_ylabel("Cantidad de Errores", fontsize=10, fontweight="bold")
            ax_err_bar.set_xticks(range(10))
            ax_err_bar.grid(axis="y", linestyle="--", alpha=0.3)
            ax_err_bar.spines["top"].set_visible(False)
            ax_err_bar.spines["right"].set_visible(False)

            for b in barras_err:
                val = b.get_height()
                if val > 0:
                    ax_err_bar.annotate(
                        f"{int(val)}",
                        xy=(b.get_x() + b.get_width() / 2, val),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                        fontweight="bold",
                    )
            fig_err_bar.tight_layout()
            st.pyplot(fig_err_bar)

        with col_g2:
            st.subheader("Confusiones Más Frecuentes")
            df_confusiones = pd.DataFrame({
                "Dígito Real": y_real_err,
                "Dígito Predicho": y_pred_err,
            })
            tabla_top_conf = (
                df_confusiones.groupby(["Dígito Real", "Dígito Predicho"])
                .size()
                .reset_index(name="Casos")
                .sort_values(by="Casos", ascending=False)
            )
            st.dataframe(tabla_top_conf.head(8), width="stretch", hide_index=True)

        st.markdown("---")

        # Filtros interactivos para la galería
        st.subheader("Galería de Imágenes Mal Clasificadas")
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])

        with f_col1:
            opciones_dig_real = ["Todos"] + sorted(list(set(y_real_err)))
            filtro_real = st.selectbox(
                "Filtrar por Dígito Real:",
                opciones_dig_real,
                index=0,
            )

        with f_col2:
            opciones_dig_pred = ["Todos"] + sorted(list(set(y_pred_err)))
            filtro_pred = st.selectbox(
                "Filtrar por Dígito Predicho:",
                opciones_dig_pred,
                index=0,
            )

        with f_col3:
            max_a_mostrar = st.select_slider(
                "Cantidad máxima a mostrar:",
                options=[8, 12, 16, 24, 32, 48],
                value=16,
            )

        # Aplicación de filtros
        filtro_mascara = np.ones(len(idx_err), dtype=bool)
        if filtro_real != "Todos":
            filtro_mascara &= (y_real_err == filtro_real)
        if filtro_pred != "Todos":
            filtro_mascara &= (y_pred_err == filtro_pred)

        X_filtrado = X_err[filtro_mascara]
        y_real_filtrado = y_real_err[filtro_mascara]
        y_pred_filtrado = y_pred_err[filtro_mascara]
        p_pred_filtrado = p_pred_err[filtro_mascara]
        p_real_filtrado = p_real_err[filtro_mascara]

        st.caption(f"Mostrando {min(len(y_real_filtrado), max_a_mostrar)} de {len(y_real_filtrado)} caso(s) seleccionado(s).")

        if len(y_real_filtrado) == 0:
            st.info("No se encontraron errores con los filtros seleccionados.")
        else:
            fig_galeria = fig_casos_mal_clasificados(
                X_filtrado,
                y_real_filtrado,
                y_pred_filtrado,
                p_pred_filtrado,
                p_real_filtrado,
                max_items=max_a_mostrar,
                cols=4,
            )
            st.pyplot(fig_galeria)

            # Detalle en tabla desplegable
            with st.expander("📋 Ver lista detallada de casos filtrados"):
                df_detalle_err = pd.DataFrame({
                    "Dígito Real": y_real_filtrado,
                    "Predicción Errónea": y_pred_filtrado,
                    "Confianza en Error": [f"{p*100:.1f}%" for p in p_pred_filtrado],
                    "Probabilidad Dígito Real": [f"{p*100:.1f}%" for p in p_real_filtrado],
                })
                st.dataframe(df_detalle_err, width="stretch")

st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "Esta configuración y sus resultados se mantendrán activos hasta que modifique los parámetros "
    "y vuelva a hacer clic en **“🚀 Entrenar y Evaluar Modelo”**."
)
