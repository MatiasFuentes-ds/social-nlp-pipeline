"""
app/dashboard.py

Dashboard interactivo tipo SPA construido con Streamlit para visualizar
las métricas de análisis de sentimiento y temáticas extraídas de comentarios
de YouTube. Consume la API RESTful (FastAPI) y soporta ingesta directa
desde URL de YouTube o archivo JSON.

Uso:
    streamlit run app/dashboard.py
"""

import json

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página — debe ser la PRIMERA llamada a Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NLP Social Media Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/MatiasFuentes-ds/social-nlp-pipeline",
        "About": "Dashboard de análisis NLP para comentarios de YouTube.",
    },
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/api"

SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral":  "#95A5A6",
}

# ---------------------------------------------------------------------------
# CSS global — modo oscuro empresarial
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* ── Reset y fondo base ─────────────────────────────────────────── */
    .stApp { background-color: #0f1117; }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Tipografía global ──────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        color: #e0e0e0;
    }

    /* ── Sidebar ────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* ── Tarjetas de KPI ────────────────────────────────────────────── */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1c2333 0%, #161b22 100%);
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.5);
    }
    div[data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* ── Secciones con tarjeta ──────────────────────────────────────── */
    .card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.35);
    }

    /* ── Títulos de sección ─────────────────────────────────────────── */
    h2 { color: #f0f6fc !important; font-weight: 700 !important; }
    h3 { color: #c9d1d9 !important; font-weight: 600 !important; }

    /* ── Dividers ───────────────────────────────────────────────────── */
    hr { border-color: #21262d !important; margin: 1.2rem 0; }

    /* ── Estado Cero ────────────────────────────────────────────────── */
    .zero-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px dashed #30363d;
        border-radius: 18px;
        margin-top: 1rem;
    }
    .zero-state h2 { font-size: 1.6rem; color: #f0f6fc !important; margin-bottom: 0.5rem; }
    .zero-state p  { color: #8b949e; max-width: 520px; font-size: 0.95rem; line-height: 1.6; }

    /* ── Botones primarios ──────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        border: 1px solid #2ea043 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em;
        transition: filter 0.2s;
    }
    .stButton > button[kind="primary"]:hover { filter: brightness(1.15); }

    /* ── Botón secundario (refrescar) ───────────────────────────────── */
    .stButton > button[kind="secondary"] {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        border-radius: 8px !important;
    }

    /* ── Inputs ─────────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stFileUploader {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #c9d1d9 !important;
    }

    /* ── Dataframe ──────────────────────────────────────────────────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── Badge de status ────────────────────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        background: #1f4a2e;
        color: #2ea043;
        border: 1px solid #2ea043;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Funciones de consumo de API con caché
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_kpis() -> dict | None:
    """Obtiene los KPIs globales de sentimiento desde el endpoint /kpis.

    Returns:
        Diccionario con conteos por categoría, o ``None`` si hay error.
    """
    try:
        r = requests.get(f"{API_URL}/kpis", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_topics() -> list[dict] | None:
    """Obtiene la distribución de temáticas con desglose de sentimiento.

    Returns:
        Lista de dicts por temática, o ``None`` si hay error.
    """
    try:
        r = requests.get(f"{API_URL}/topics", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_comments(limit: int = 300, offset: int = 0) -> dict | None:
    """Obtiene la lista paginada de comentarios con predicciones NLP.

    Args:
        limit: Número máximo de comentarios a recuperar.
        offset: Registros a saltar (paginación).

    Returns:
        Dict con ``total`` e ``items``, o ``None`` si hay error.
    """
    try:
        r = requests.get(
            f"{API_URL}/comments",
            params={"limit": limit, "offset": offset},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers de gráficos
# ---------------------------------------------------------------------------

def build_donut_chart(kpis: dict) -> px.pie:
    """Construye un gráfico de dona con la distribución de sentimientos."""
    df = pd.DataFrame({
        "Sentimiento": ["Positivo", "Negativo", "Neutro"],
        "Cantidad":    [kpis["positive_count"], kpis["negative_count"], kpis["neutral_count"]],
        "ckey":        ["positive", "negative", "neutral"],
    })
    fig = px.pie(
        df,
        names="Sentimiento",
        values="Cantidad",
        hole=0.45,
        color="ckey",
        color_discrete_map={k: v for k, v in SENTIMENT_COLORS.items()},
        title="Distribución de Sentimientos",
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Cantidad: %{value}<br>%{percent}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9", family="Inter, system-ui, sans-serif"),
        legend=dict(orientation="h", y=-0.12, font_size=12),
        title_font_size=15,
        margin=dict(t=55, b=30, l=20, r=20),
    )
    return fig


def build_topics_bar_chart(topics_data: list[dict]) -> px.bar:
    """Construye un gráfico de barras horizontales apiladas por temática."""
    rows = []
    for topic in topics_data:
        label = topic["topic_label"]
        bd = topic["sentiment_breakdown"]
        for sent, ckey in [("Positivo", "positive"), ("Negativo", "negative"), ("Neutro", "neutral")]:
            rows.append({"Tema": label, "Sentimiento": sent, "ckey": ckey, "Cantidad": bd[ckey]})

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Cantidad",
        y="Tema",
        color="ckey",
        color_discrete_map={k: v for k, v in SENTIMENT_COLORS.items()},
        orientation="h",
        barmode="stack",
        title="Comentarios por Temática y Sentimiento",
        labels={"Cantidad": "N° Comentarios", "Tema": ""},
        hover_data={"ckey": False, "Sentimiento": True},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9", family="Inter, system-ui, sans-serif"),
        legend_title_text="Sentimiento",
        legend=dict(orientation="h", y=-0.18, font_size=12),
        title_font_size=15,
        margin=dict(t=55, b=30, l=20, r=20),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Renderiza el panel lateral con branding y controles de ingesta."""
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem">
                <img src="https://img.icons8.com/fluency/48/youtube-play.png" width="36"/>
                <span style="font-size:1.2rem;font-weight:700;color:#f0f6fc">NLP Analyzer</span>
            </div>
            <p style="color:#8b949e;font-size:0.82rem;margin-top:0;line-height:1.5">
                Pipeline de análisis de sentimiento<br>y temáticas para YouTube.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:0.5rem">
                <span style="background:#1f2d3d;color:#58a6ff;border:1px solid #1f6feb;
                             border-radius:12px;padding:2px 10px;font-size:0.72rem">RoBERTa</span>
                <span style="background:#1c2a1c;color:#2ea043;border:1px solid #2ea043;
                             border-radius:12px;padding:2px 10px;font-size:0.72rem">BART</span>
                <span style="background:#2a1f2d;color:#a371f7;border:1px solid #6e40c9;
                             border-radius:12px;padding:2px 10px;font-size:0.72rem">SQLite</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Configuración de Datos ────────────────────────────────────────
        st.markdown("### ⚙️ Configuración de Datos")

        metodo = st.radio(
            "Método de ingesta",
            options=["🔗 Enlace de YouTube", "📂 Cargar archivo JSON"],
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Opción 1: URL de YouTube ──────────────────────────────────────
        if metodo == "🔗 Enlace de YouTube":
            url_input = st.text_input(
                "URL del video",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="visible",
            )
            max_pages = st.slider(
                "Páginas a extraer",
                min_value=1, max_value=10, value=3,
                help="Cada página contiene ~100 comentarios. Más páginas = mayor tiempo.",
            )
            procesar = st.button(
                "▶ Procesar Video",
                type="primary",
                use_container_width=True,
            )

            if procesar:
                if not url_input.strip():
                    st.warning("⚠️ Ingresa una URL válida de YouTube.")
                else:
                    with st.spinner("Limpiando base de datos y extrayendo comentarios..."):
                        try:
                            resp = requests.post(
                                f"{API_URL}/analyze/url",
                                json={"url": url_input.strip(), "max_pages": max_pages},
                                timeout=600,  # el pipeline puede tardar varios minutos
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                st.success(
                                    f"✅ ¡Análisis completado! "
                                    f"{data['comments_processed']} comentarios procesados."
                                )
                                st.cache_data.clear()
                                st.rerun()
                            elif resp.status_code == 400:
                                st.error(f"❌ URL inválida: {resp.json().get('detail', '')}")
                            elif resp.status_code == 404:
                                st.warning("⚠️ Video sin comentarios o ID inválido.")
                            else:
                                st.error(
                                    f"❌ Error {resp.status_code}: "
                                    f"{resp.json().get('detail', 'Error desconocido')}"
                                )
                        except requests.exceptions.Timeout:
                            st.error(
                                "⏱️ La solicitud tardó demasiado. El pipeline sigue corriendo "
                                "en segundo plano — espera unos minutos y refresca."
                            )
                        except requests.exceptions.ConnectionError:
                            st.error("⚠️ No se puede conectar a la API en `localhost:8000`.")

        # ── Opción 2: Archivo JSON ────────────────────────────────────────
        else:
            uploaded = st.file_uploader(
                "Sube tu archivo de comentarios",
                type=["json"],
                help="Formato esperado: lista de dicts con comment_id, text, author, etc.",
            )
            if uploaded is not None:
                with st.spinner("Procesando archivo JSON..."):
                    try:
                        comments = json.load(uploaded)
                        if not isinstance(comments, list) or len(comments) == 0:
                            st.error("❌ El archivo debe ser una lista JSON no vacía.")
                        else:
                            resp = requests.post(
                                f"{API_URL}/analyze/url",
                                json={"url": "https://youtu.be/local_json", "max_pages": 1},
                                timeout=600,
                            )
                            # Nota: el flujo JSON completo requiere endpoint dedicado.
                            # Por ahora se notifica al usuario del estado actual.
                            st.info(
                                f"📋 Archivo cargado con {len(comments)} comentarios. "
                                "El endpoint de ingesta directa por JSON estará disponible "
                                "en la próxima versión de la API."
                            )
                    except json.JSONDecodeError:
                        st.error("❌ El archivo no es un JSON válido.")
                    except Exception as exc:
                        st.error(f"❌ Error inesperado: {exc}")

        st.divider()

        # ── Botón refrescar ───────────────────────────────────────────────
        if st.button("🔄 Refrescar Datos", use_container_width=True, type="secondary"):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            "<p style='color:#484f58;font-size:0.75rem;margin-top:1rem'>"
            "API: <code>localhost:8000</code><br>Caché TTL: 300s</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Estado cero — cuando no hay datos en la API
# ---------------------------------------------------------------------------

def render_zero_state() -> None:
    """Muestra una pantalla de bienvenida elegante cuando no hay datos cargados."""
    st.markdown(
        """
        <div class="zero-state">
            <div style="font-size:4rem;margin-bottom:1rem">🎯</div>
            <h2>Bienvenido a NLP Social Media Analyzer</h2>
            <p>
                Aún no hay datos cargados. Para comenzar, ingresa la URL de un video
                de YouTube en el panel izquierdo y presiona <strong>"Procesar Video"</strong>.
                <br><br>
                El pipeline extraerá los comentarios automáticamente y aplicará modelos
                de análisis de sentimiento (<strong>RoBERTa</strong>) y clasificación
                temática (<strong>BART zero-shot</strong>) en tiempo real.
            </p>
            <div style="display:flex;gap:1rem;margin-top:1.5rem;flex-wrap:wrap;justify-content:center">
                <div style="background:#1c2333;border:1px solid #21262d;border-radius:10px;
                            padding:0.8rem 1.2rem;text-align:center;min-width:130px">
                    <div style="font-size:1.5rem">💬</div>
                    <div style="color:#8b949e;font-size:0.78rem;margin-top:4px">Comentarios</div>
                    <div style="color:#f0f6fc;font-weight:700;font-size:1.1rem">—</div>
                </div>
                <div style="background:#1c2333;border:1px solid #21262d;border-radius:10px;
                            padding:0.8rem 1.2rem;text-align:center;min-width:130px">
                    <div style="font-size:1.5rem">🤖</div>
                    <div style="color:#8b949e;font-size:0.78rem;margin-top:4px">Modelos activos</div>
                    <div style="color:#f0f6fc;font-weight:700;font-size:1.1rem">2</div>
                </div>
                <div style="background:#1c2333;border:1px solid #21262d;border-radius:10px;
                            padding:0.8rem 1.2rem;text-align:center;min-width:130px">
                    <div style="font-size:1.5rem">🏷️</div>
                    <div style="color:#8b949e;font-size:0.78rem;margin-top:4px">Temáticas</div>
                    <div style="color:#f0f6fc;font-weight:700;font-size:1.1rem">5</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Secciones del dashboard
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Renderiza el encabezado principal con título y badge de estado."""
    col_title, col_badge = st.columns([6, 1])
    with col_title:
        st.markdown(
            "<h1 style='color:#f0f6fc;font-size:1.9rem;font-weight:800;margin-bottom:0'>"
            "🎯 NLP Social Media Analyzer</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#8b949e;font-size:0.9rem;margin-top:0.2rem'>"
            "Análisis de sentimiento y temáticas en comentarios de YouTube · "
            "Modelos: <code>cardiffnlp/twitter-roberta-base-sentiment-latest</code> + "
            "<code>facebook/bart-large-mnli</code></p>",
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            "<div style='text-align:right;padding-top:0.6rem'>"
            "<span class='status-badge'>● API ONLINE</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()


def render_kpis(kpis: dict) -> None:
    """Renderiza las 4 tarjetas de métricas clave en una fila."""
    total = kpis["total_comments"]
    pos   = kpis["positive_count"]
    neg   = kpis["negative_count"]
    neu   = kpis["neutral_count"]
    pct   = lambda n: f"{n / total * 100:.1f}%" if total > 0 else "—"

    st.markdown("#### 📊 Métricas Globales")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("💬 Total Comentarios", f"{total:,}")
    with c2:
        st.metric("✅ Positivos", f"{pos:,}", delta=pct(pos), delta_color="normal")
    with c3:
        st.metric("❌ Negativos", f"{neg:,}", delta=pct(neg), delta_color="inverse")
    with c4:
        st.metric("➖ Neutros", f"{neu:,}", delta=pct(neu), delta_color="off")


def render_charts(kpis: dict, topics_data: list[dict]) -> None:
    """Renderiza los dos gráficos principales dentro de contenedores con bordes."""
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("#### 📈 Visualizaciones")

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig_donut = build_donut_chart(kpis)
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if topics_data:
                fig_bar = build_topics_bar_chart(topics_data)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No hay datos de temáticas disponibles.")
            st.markdown("</div>", unsafe_allow_html=True)


def render_comments_table(comments_data: dict) -> None:
    """Renderiza la tabla exploratoria de comentarios con sus predicciones."""
    total = comments_data.get("total", 0)
    items = comments_data.get("items", [])

    st.divider()
    st.markdown(
        f"#### 🔍 Exploración de Comentarios "
        f"<span style='color:#8b949e;font-size:0.85rem;font-weight:400'>"
        f"({total:,} registros en total)</span>",
        unsafe_allow_html=True,
    )
    st.caption("Mostrando hasta 300 comentarios más recientes con predicciones del modelo.")

    if not items:
        st.info("No hay comentarios disponibles. Ejecuta el pipeline primero.")
        return

    df = pd.DataFrame(items)
    columnas = {
        "author":          "Autor",
        "text":            "Comentario",
        "like_count":      "Likes",
        "sentiment_label": "Sentimiento",
        "sentiment_score": "Confianza (Sent.)",
        "topic_label":     "Temática",
        "topic_score":     "Confianza (Tema)",
        "published_at":    "Publicado",
    }
    df = df[[c for c in columnas if c in df.columns]].rename(columns=columnas)
    for col in ["Confianza (Sent.)", "Confianza (Tema)"]:
        if col in df.columns:
            df[col] = df[col].round(3)

    st.dataframe(
        df,
        use_container_width=True,
        height=420,
        column_config={
            "Comentario": st.column_config.TextColumn(width="large"),
            "Likes": st.column_config.NumberColumn(format="%d ❤️"),
            "Confianza (Sent.)": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.3f"
            ),
            "Confianza (Tema)": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.3f"
            ),
        },
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Orquesta la construcción completa del dashboard.

    Flujo:
        1. Inyecta CSS global.
        2. Renderiza el sidebar con controles de ingesta.
        3. Renderiza el header.
        4. Carga datos desde la API.
        5. Si no hay datos → muestra estado cero.
        6. Si hay datos → KPIs, gráficos y tabla de exploración.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_sidebar()
    render_header()

    # ── Carga de datos ─────────────────────────────────────────────────────
    with st.spinner("Cargando datos desde la API..."):
        kpis          = fetch_kpis()
        topics_data   = fetch_topics()
        comments_data = fetch_comments(limit=300)

    # ── Estado cero: API caída o sin datos ─────────────────────────────────
    api_unreachable = kpis is None
    no_data         = kpis is not None and kpis.get("total_comments", 0) == 0

    if api_unreachable:
        st.markdown(
            "<div style='background:#1c1f26;border:1px solid #f85149;border-radius:10px;"
            "padding:1rem 1.5rem;color:#f85149;margin-bottom:1rem'>"
            "⚠️ <strong>No se puede conectar a la API.</strong> "
            "Asegúrate de que FastAPI esté corriendo en <code>localhost:8000</code>.</div>",
            unsafe_allow_html=True,
        )
        render_zero_state()
        return

    if no_data:
        render_zero_state()
        return

    # ── Dashboard con datos ────────────────────────────────────────────────
    render_kpis(kpis)
    render_charts(kpis, topics_data or [])

    if comments_data:
        render_comments_table(comments_data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
