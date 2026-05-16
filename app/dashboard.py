"""
app/dashboard.py

Dashboard interactivo tipo SPA construido con Streamlit para visualizar
las métricas de análisis de sentimiento y temáticas extraídas de comentarios
de YouTube. Consume la API RESTful (FastAPI) en tiempo real.

Uso:
    streamlit run app/dashboard.py
"""

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

# Paleta de colores consistente para sentimientos
SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#2ECC71",  # Verde
    "negative": "#E74C3C",  # Rojo
    "neutral":  "#95A5A6",  # Gris
}

# ---------------------------------------------------------------------------
# CSS personalizado — apariencia moderna y profesional
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Fondo principal */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Tarjetas de KPI */
    div[data-testid="metric-container"] {
        background-color: #1E1E2E;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Valor del KPI */
    div[data-testid="metric-container"] > label {
        color: #CDD6F4 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="metric-container"] > div {
        color: #FFFFFF !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #181825;
        border-right: 1px solid #313244;
    }

    /* Separadores */
    hr {
        border-color: #313244;
        margin: 1.5rem 0;
    }

    /* Títulos de sección */
    h2, h3 {
        color: #CDD6F4;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Funciones de consumo de API con caché
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_kpis() -> dict | None:
    """Obtiene los KPIs globales de sentimiento desde el endpoint /kpis.

    Mantiene los datos cacheados durante 5 minutos para evitar saturar
    la API en cada recarga de la interfaz.

    Returns:
        Diccionario con ``total_comments``, ``positive_count``,
        ``negative_count`` y ``neutral_count``, o ``None`` si la API
        no está disponible.
    """
    try:
        response = requests.get(f"{API_URL}/kpis", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ No se puede conectar a la API. "
            "Asegúrate de que FastAPI esté corriendo en `localhost:8000`."
        )
    except requests.exceptions.Timeout:
        st.error("⏱️ La API tardó demasiado en responder. Intenta refrescar.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"❌ Error HTTP al obtener KPIs: {exc}")
    return None


@st.cache_data(ttl=300)
def fetch_topics() -> list[dict] | None:
    """Obtiene la distribución de temáticas con desglose de sentimiento.

    Returns:
        Lista de dicts con ``topic_label``, ``count`` y
        ``sentiment_breakdown``, o ``None`` si hay un error de conexión.
    """
    try:
        response = requests.get(f"{API_URL}/topics", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ No se puede conectar a la API. "
            "Asegúrate de que FastAPI esté corriendo en `localhost:8000`."
        )
    except requests.exceptions.Timeout:
        st.error("⏱️ La API tardó demasiado en responder. Intenta refrescar.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"❌ Error HTTP al obtener Topics: {exc}")
    return None


@st.cache_data(ttl=300)
def fetch_comments(limit: int = 300, offset: int = 0) -> dict | None:
    """Obtiene la lista paginada de comentarios con sus predicciones NLP.

    Args:
        limit: Número máximo de comentarios a recuperar.
        offset: Número de registros a saltar (paginación).

    Returns:
        Dict con ``total``, ``limit``, ``offset`` e ``items`` (lista de
        comentarios enriquecidos), o ``None`` si hay un error de conexión.
    """
    try:
        response = requests.get(
            f"{API_URL}/comments",
            params={"limit": limit, "offset": offset},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ No se puede conectar a la API. "
            "Asegúrate de que FastAPI esté corriendo en `localhost:8000`."
        )
    except requests.exceptions.Timeout:
        st.error("⏱️ La API tardó demasiado en responder. Intenta refrescar.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"❌ Error HTTP al obtener Comentarios: {exc}")
    return None


# ---------------------------------------------------------------------------
# Helpers de construcción de gráficos
# ---------------------------------------------------------------------------

def build_donut_chart(kpis: dict) -> px.pie:
    """Construye un gráfico de dona con la distribución de sentimientos.

    Args:
        kpis: Diccionario con los conteos por categoría de sentimiento.

    Returns:
        Figura de Plotly Express lista para renderizar con ``st.plotly_chart``.
    """
    df = pd.DataFrame(
        {
            "Sentimiento": ["Positivo", "Negativo", "Neutro"],
            "Cantidad": [
                kpis["positive_count"],
                kpis["negative_count"],
                kpis["neutral_count"],
            ],
            "color_key": ["positive", "negative", "neutral"],
        }
    )
    fig = px.pie(
        df,
        names="Sentimiento",
        values="Cantidad",
        hole=0.4,
        color="color_key",
        color_discrete_map={
            "positive": SENTIMENT_COLORS["positive"],
            "negative": SENTIMENT_COLORS["negative"],
            "neutral":  SENTIMENT_COLORS["neutral"],
        },
        title="Distribución de Sentimientos",
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CDD6F4",
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=True,
    )
    return fig


def build_topics_bar_chart(topics_data: list[dict]) -> px.bar:
    """Construye un gráfico de barras horizontales apiladas por temática.

    Args:
        topics_data: Lista de dicts provenientes del endpoint /topics,
            cada uno con ``topic_label``, ``count`` y
            ``sentiment_breakdown``.

    Returns:
        Figura de Plotly Express lista para renderizar con ``st.plotly_chart``.
    """
    rows = []
    for topic in topics_data:
        label = topic["topic_label"]
        breakdown = topic["sentiment_breakdown"]
        rows.append({"Tema": label, "Sentimiento": "Positivo", "color_key": "positive", "Cantidad": breakdown["positive"]})
        rows.append({"Tema": label, "Sentimiento": "Negativo", "color_key": "negative", "Cantidad": breakdown["negative"]})
        rows.append({"Tema": label, "Sentimiento": "Neutro",   "color_key": "neutral",  "Cantidad": breakdown["neutral"]})

    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="Cantidad",
        y="Tema",
        color="color_key",
        color_discrete_map={
            "positive": SENTIMENT_COLORS["positive"],
            "negative": SENTIMENT_COLORS["negative"],
            "neutral":  SENTIMENT_COLORS["neutral"],
        },
        orientation="h",
        barmode="stack",
        title="Comentarios por Temática y Sentimiento",
        labels={"Cantidad": "N° Comentarios", "Tema": ""},
        hover_data={"color_key": False, "Sentimiento": True},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CDD6F4",
        legend_title_text="Sentimiento",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis=dict(gridcolor="#313244"),
        yaxis=dict(gridcolor="#313244"),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Renderiza el contenido del panel lateral (sidebar).

    Incluye información del proyecto y un botón para limpiar el caché
    y recargar los datos desde la API.
    """
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/youtube-play.png",
            width=64,
        )
        st.title("NLP Analyzer")
        st.markdown(
            """
            **Pipeline de análisis NLP** para comentarios de YouTube.

            - 🤖 Modelos: `RoBERTa` + `BART`
            - 🗄️ Almacenamiento: `SQLite`
            - ⚡ API: `FastAPI`
            """
        )
        st.divider()

        if st.button("🔄 Refrescar Datos", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption("API conectada en `localhost:8000`")
        st.caption("Caché: 5 minutos (TTL=300s)")


# ---------------------------------------------------------------------------
# Secciones del dashboard
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Renderiza el encabezado principal del dashboard."""
    st.title("🎯 NLP Social Media Analyzer")
    st.markdown(
        """
        Dashboard de análisis de sentimiento y temáticas extraído de comentarios
        de YouTube usando modelos de Hugging Face (`cardiffnlp/twitter-roberta-base-sentiment-latest`
        y `facebook/bart-large-mnli`).
        """
    )
    st.divider()


def render_kpis(kpis: dict) -> None:
    """Renderiza las 4 tarjetas de métricas clave (KPIs).

    Args:
        kpis: Diccionario con los conteos por categoría de sentimiento.
    """
    st.subheader("📊 Métricas Globales")
    col1, col2, col3, col4 = st.columns(4)

    total = kpis["total_comments"]
    pos   = kpis["positive_count"]
    neg   = kpis["negative_count"]
    neu   = kpis["neutral_count"]

    pct = lambda n: f"{n / total * 100:.1f}%" if total > 0 else "—"

    with col1:
        st.metric("💬 Total Comentarios", f"{total:,}")
    with col2:
        st.metric("✅ Positivos", f"{pos:,}", delta=pct(pos), delta_color="normal")
    with col3:
        st.metric("❌ Negativos", f"{neg:,}", delta=pct(neg), delta_color="inverse")
    with col4:
        st.metric("➖ Neutros", f"{neu:,}", delta=pct(neu), delta_color="off")


def render_charts(kpis: dict, topics_data: list[dict]) -> None:
    """Renderiza los dos gráficos principales en layout de dos columnas.

    Args:
        kpis: Diccionario con los conteos de sentimiento para el donut chart.
        topics_data: Lista de temáticas para el gráfico de barras apiladas.
    """
    st.divider()
    st.subheader("📈 Visualizaciones")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_donut = build_donut_chart(kpis)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        if topics_data:
            fig_bar = build_topics_bar_chart(topics_data)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No hay datos de temáticas disponibles.")


def render_comments_table(comments_data: dict) -> None:
    """Renderiza la tabla exploratoria de comentarios con sus predicciones NLP.

    Args:
        comments_data: Dict con ``total`` e ``items`` proveniente de la API.
    """
    st.divider()
    total = comments_data.get("total", 0)
    items = comments_data.get("items", [])

    st.subheader(f"🔍 Exploración de Comentarios ({total:,} en total)")
    st.caption("Mostrando hasta 300 comentarios más recientes con sus predicciones del modelo.")

    if not items:
        st.info("No hay comentarios disponibles. Ejecuta el pipeline primero.")
        return

    df = pd.DataFrame(items)

    # Seleccionar y renombrar columnas relevantes
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

    # Redondear scores
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
    """Función principal que orquesta la construcción completa del dashboard.

    Inyecta el CSS personalizado, renderiza el sidebar, carga los datos
    desde la API y construye cada sección de la interfaz en orden.
    """
    # Estilos globales
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Header
    render_header()

    # Carga de datos con spinners
    with st.spinner("Cargando métricas desde la API..."):
        kpis = fetch_kpis()
        topics_data = fetch_topics()
        comments_data = fetch_comments(limit=300)

    # Si la API no responde, los errores ya se mostraron en las funciones fetch_*
    if kpis is None:
        st.warning(
            "No se pudieron cargar los datos. "
            "Verifica que la API esté corriendo y presiona **Refrescar Datos**."
        )
        return

    # Sección KPIs
    render_kpis(kpis)

    # Sección gráficos
    render_charts(kpis, topics_data or [])

    # Sección tabla exploratoria
    if comments_data:
        render_comments_table(comments_data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

