"""
app/dashboard.py

Aplicación Streamlit con flujo guiado de tres estados:
1. Landing: ingreso de URL de YouTube.
2. Processing: pantalla de espera mientras corre el pipeline.
3. Results: resumen narrativo + visualización principal + detalle.

Uso:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YouTube Comment Section Analyzer",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000/api")
REQUEST_TIMEOUT_ANALYZE = 600
REQUEST_TIMEOUT_READ = 20

SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#36C275",
    "neutral": "#A0A7B4",
    "negative": "#F05D5E",
}

# ---------------------------------------------------------------------------
# CSS global
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: #f7f4ee;
        color: #121212;
    }

    html, body, [class*="css"]  {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 0.75rem;
        padding-bottom: 2.5rem;
    }

    .hero-wrap {
        min-height: auto;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding-top: 0.2rem;
    }

    .hero-card {
        width: 100%;
        max-width: 760px;
        text-align: center;
        padding: 0.5rem 1rem 1.25rem 1rem;
    }

    .hero-kicker {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #7b766f;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }

    .hero-title {
        font-size: clamp(2.6rem, 6vw, 5.5rem);
        line-height: 0.95;
        letter-spacing: -0.04em;
        color: #111111;
        font-weight: 800;
        margin-bottom: 0.9rem;
    }

    .hero-subtitle {
        max-width: 620px;
        margin: 0 auto 1.4rem auto;
        color: #655f58;
        font-size: 1.02rem;
        line-height: 1.7;
    }

    .small-link {
        text-align: center;
        color: #6d675f;
        font-size: 0.92rem;
        margin-top: 1rem;
    }

    .eta-chip {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: #ece7de;
        color: #5f5a53;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        border: 1px solid #ddd6ca;
    }

    .processing-wrap {
        width: 100%;
        max-width: 760px;
        margin: 0 auto;
        padding-top: 0.35rem;
    }

    .processing-title {
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.02;
        letter-spacing: -0.04em;
        color: #111111;
        font-weight: 800;
        margin-bottom: 0.7rem;
        text-align: center;
    }

    .processing-subtitle {
        text-align: center;
        color: #6a645c;
        max-width: 680px;
        margin: 0 auto 1.3rem auto;
        line-height: 1.7;
    }

    .video-title-chip {
        text-align: center;
        color: #5f5a53;
        font-size: 1rem;
        margin-bottom: 0.85rem;
        font-weight: 600;
    }

    .step-card {
        background: rgba(255,255,255,0.62);
        border: 1px solid #ddd5c7;
        border-radius: 18px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
    }

    .step-title {
        font-size: 1rem;
        font-weight: 700;
        color: #151515;
        margin-bottom: 0.2rem;
    }

    .step-copy {
        font-size: 0.95rem;
        color: #6f685f;
        line-height: 1.55;
    }

    .result-video-title {
        font-size: 1.05rem;
        color: #5c574f;
        margin-bottom: 0.85rem;
        font-weight: 600;
    }

    .result-hero {
        margin-top: 0.5rem;
        margin-bottom: 1.7rem;
    }

    .result-big {
        font-size: clamp(2.4rem, 6vw, 5rem);
        font-weight: 800;
        color: #111111;
        line-height: 0.95;
        letter-spacing: -0.045em;
        margin-bottom: 0.55rem;
    }

    .result-summary {
        font-size: clamp(1.1rem, 2vw, 1.5rem);
        color: #5c564f;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    .soft-panel {
        background: rgba(255,255,255,0.56);
        border: 1px solid #ddd5c7;
        border-radius: 22px;
        padding: 1.2rem 1.2rem;
    }

    .section-label {
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #8a8379;
        margin-bottom: 0.7rem;
        font-weight: 700;
    }

    .stat-mini {
        background: rgba(255,255,255,0.58);
        border: 1px solid #ded7cb;
        border-radius: 18px;
        padding: 1rem;
        min-height: 115px;
    }

    .stat-mini-label {
        color: #7c756d;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.55rem;
    }

    .stat-mini-value {
        color: #111111;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1;
    }

    .stat-mini-copy {
        color: #726c63;
        font-size: 0.9rem;
        margin-top: 0.45rem;
        line-height: 1.45;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.75) !important;
        border: 1px solid #d8d0c2 !important;
        border-radius: 18px !important;
        color: #111 !important;
        padding-top: 0.8rem !important;
        padding-bottom: 0.8rem !important;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stTextArea"] label {
        color: #5f5a53 !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        border-radius: 999px !important;
        padding: 0.75rem 1.4rem !important;
        border: 1px solid #111 !important;
        background: #111 !important;
        color: #fff !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #2a2a2a !important;
        border-color: #2a2a2a !important;
        color: #fff !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #ded7ca;
        border-radius: 18px;
        overflow: hidden;
    }

    .empty-note {
        background: rgba(255,255,255,0.58);
        border: 1px solid #ddd5c8;
        border-radius: 20px;
        padding: 1.2rem;
        color: #6a635b;
        line-height: 1.6;
    }
</style>
"""

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Inicializa el estado de navegación y datos de la app."""
    defaults = {
        "page": "landing",
        "video_url": "",
        "max_pages": 3,
        "candidate_labels_raw": "Music, Controversy, Fashion/Yeezy, Politics, Religion",
        "candidate_labels": ["Music", "Controversy", "Fashion/Yeezy", "Politics", "Religion"],
        "last_result": None,
        "last_error": None,
        "last_video_title": "",
        "last_video_id": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_kpis() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}/kpis", timeout=REQUEST_TIMEOUT_READ)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_topics() -> list[dict[str, Any]] | None:
    try:
        response = requests.get(f"{API_URL}/topics", timeout=REQUEST_TIMEOUT_READ)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_comments(limit: int = 200, offset: int = 0) -> dict[str, Any] | None:
    try:
        response = requests.get(
            f"{API_URL}/comments",
            params={"limit": limit, "offset": offset},
            timeout=REQUEST_TIMEOUT_READ,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def extract_video_id(url: str) -> str | None:
    """Extrae el video_id desde URLs comunes de YouTube."""
    pattern = re.compile(
        r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
    )
    match = pattern.search(url.strip())
    return match.group(1) if match else None


def parse_candidate_labels(raw_text: str) -> list[str]:
    """Convierte un texto separado por comas en etiquetas limpias."""
    labels = [item.strip() for item in raw_text.split(",") if item.strip()]
    return labels


def fetch_video_title(video_id: str) -> str:
    """Consulta el título real del video desde el endpoint o devuelve fallback elegante."""
    if not video_id:
        return "YouTube Video"

    try:
        response = requests.get(
            f"{API_URL}/comments",
            params={"limit": 1, "offset": 0},
            timeout=REQUEST_TIMEOUT_READ,
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items", [])
        if items:
            return f"Video ID: {items[0].get('video_id', video_id)}"
    except Exception:
        pass

    return f"Video ID: {video_id}"


def get_dominant_sentiment(kpis: dict[str, Any]) -> tuple[str, int]:
    mapping = {
        "positive": kpis.get("positive_count", 0),
        "neutral": kpis.get("neutral_count", 0),
        "negative": kpis.get("negative_count", 0),
    }
    label = max(mapping, key=mapping.get)
    return label, mapping[label]


def sentiment_summary_text(kpis: dict[str, Any]) -> str:
    dominant, count = get_dominant_sentiment(kpis)
    total = max(kpis.get("total_comments", 0), 1)
    pct = round((count / total) * 100)
    label_map = {
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative",
    }
    return f"Most of them are {label_map[dominant]} — around {pct}% of the analyzed comments."


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def build_sentiment_bar_chart(kpis: dict[str, Any]):
    df = pd.DataFrame(
        {
            "Sentiment": ["Pos", "Neu", "Neg"],
            "value": [
                kpis.get("positive_count", 0),
                kpis.get("neutral_count", 0),
                kpis.get("negative_count", 0),
            ],
            "key": ["positive", "neutral", "negative"],
        }
    )
    fig = px.bar(
        df,
        x="Sentiment",
        y="value",
        color="key",
        color_discrete_map=SENTIMENT_COLORS,
        text_auto=True,
        title="Sentiment distribution",
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Comments: %{y}<extra></extra>",
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#171717", family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(title="", showgrid=False, zeroline=False),
        yaxis=dict(title="", gridcolor="#e7dfd2", zeroline=False),
        title_font_size=16,
    )
    return fig


def build_topics_chart(topics_data: list[dict[str, Any]]):
    if not topics_data:
        return None
    df = pd.DataFrame(
        {
            "Topic": [topic["topic_label"] for topic in topics_data],
            "Comments": [topic["count"] for topic in topics_data],
        }
    ).sort_values("Comments", ascending=True)
    fig = px.bar(
        df,
        x="Comments",
        y="Topic",
        orientation="h",
        text_auto=True,
        title="Topic distribution",
    )
    fig.update_traces(marker_color="#111111", marker_line_width=0)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#171717", family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(title="", gridcolor="#e7dfd2", zeroline=False),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
        title_font_size=16,
    )
    return fig


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------
def render_customization_controls() -> None:
    """Renderiza opciones de personalización para el análisis."""
    with st.expander("Customize analysis", expanded=False):
        st.session_state.max_pages = st.number_input(
            "How many pages should be analyzed?",
            min_value=1,
            max_value=10,
            value=int(st.session_state.max_pages),
            step=1,
        )
        st.caption("1 page = aproximadamente 100 comentarios.")

        labels_raw = st.text_area(
            "Custom topic categories",
            value=st.session_state.candidate_labels_raw,
            height=110,
            help="Ingresa categorías separadas por comas. Ejemplo: Music, Lyrics, Scandal, Performance",
        )
        st.session_state.candidate_labels_raw = labels_raw
        parsed = parse_candidate_labels(labels_raw)
        if parsed:
            st.session_state.candidate_labels = parsed
            st.caption(f"Active labels: {', '.join(parsed)}")
        else:
            st.warning("Please enter at least one valid category.")


def render_landing() -> None:
    st.markdown('<div class="hero-wrap"><div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<div class="hero-kicker">YouTube Link</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">YouTube Comment<br>Section Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">'
        'Paste a YouTube video link and get a clean summary of how people reacted: '
        'sentiment distribution, dominant opinion, key themes, and a structured view '
        'of the comments behind the analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    url = st.text_input(
        "YouTube Link",
        value=st.session_state.video_url,
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    st.session_state.video_url = url

    render_customization_controls()

    col_a, col_b, col_c = st.columns([1.2, 1, 1.2])
    with col_b:
        if st.button("Analyze", use_container_width=True):
            if not url.strip():
                st.session_state.last_error = "Please paste a YouTube URL first."
            elif not extract_video_id(url):
                st.session_state.last_error = "The URL does not look like a valid YouTube video link."
            elif not st.session_state.candidate_labels:
                st.session_state.last_error = "Please define at least one valid topic category."
            else:
                st.session_state.last_error = None
                st.session_state.page = "processing"
                st.rerun()

    if st.session_state.last_error:
        st.warning(st.session_state.last_error)

    st.markdown('<div class="small-link">ABOUT</div>', unsafe_allow_html=True)
    with st.expander("About this app", expanded=False):
        st.write(
            "This app runs an end-to-end NLP pipeline over YouTube comments, combining "
            "sentiment analysis and zero-shot topic classification to turn raw reactions "
            "into an interpretable summary."
        )
        st.markdown(
            'GitHub profile: [MatiasFuentes-ds](https://github.com/MatiasFuentes-ds)',
            unsafe_allow_html=False,
        )

    st.markdown('</div></div>', unsafe_allow_html=True)


def render_processing() -> None:
    st.markdown('<div class="processing-wrap">', unsafe_allow_html=True)

    video_id = extract_video_id(st.session_state.video_url) or ""
    video_title = st.session_state.get("last_video_title", "") or f"Video ID: {video_id}"
    st.session_state.last_video_title = video_title
    st.session_state.last_video_id = video_id

    st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
    st.markdown('<div class="eta-chip">estimated: 2 minutes</div>', unsafe_allow_html=True)
    st.markdown('<div class="video-title-chip">{}</div>'.format(video_title), unsafe_allow_html=True)
    st.markdown('<div class="processing-title">Analyzing your video</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="processing-subtitle">'
        'We are cleaning previous data, extracting comments from YouTube, and running '
        'the NLP models to produce a fresh analysis for this video.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    step_1 = st.empty()
    step_2 = st.empty()
    step_3 = st.empty()
    step_4 = st.empty()

    step_1.info("1. Resetting the local dataset so the dashboard shows only one clean analysis at a time.")
    step_2.info("2. Preparing extraction of YouTube comments from the selected video.")
    step_3.info("3. Running sentiment analysis and topic classification over the retrieved comments.")
    step_4.info("4. Refreshing the final view with updated metrics, chart data, and comments.")

    with st.spinner("Limpiando base de datos y extrayendo comentarios..."):
        try:
            step_1.success("1. Local dataset cleared successfully.")
            step_2.success(
                f"2. Extracting up to {st.session_state.max_pages} page(s) of comments "
                f"(~{int(st.session_state.max_pages) * 100} comments maximum)."
            )
            step_3.info(
                "3. Running sentiment model + zero-shot topic classification with labels: "
                + ", ".join(st.session_state.candidate_labels)
            )

            response = requests.post(
                f"{API_URL}/analyze/url",
                json={
                    "url": st.session_state.video_url,
                    "max_pages": int(st.session_state.max_pages),
                    "candidate_labels": st.session_state.candidate_labels,
                },
                timeout=REQUEST_TIMEOUT_ANALYZE,
            )

            if response.status_code == 200:
                step_3.success(
                    "3. NLP inference completed successfully using the selected categories."
                )
                step_4.success("4. Dashboard data updated. Redirecting to results view...")
                st.session_state.last_result = response.json()
                
                api_title = response.json().get("video_title", "")
                if api_title:
                    st.session_state.last_video_title = api_title

                st.session_state.last_error = None
                st.cache_data.clear()
                st.session_state.page = "results"
                st.rerun()
            else:
                try:
                    detail = response.json().get("detail", "Unexpected API error.")
                except Exception:
                    detail = response.text or "Unexpected API error."
                st.session_state.last_error = detail
                st.session_state.page = "landing"
                st.rerun()
        except requests.exceptions.ConnectionError:
            st.session_state.last_error = "Could not connect to the API. Make sure FastAPI is running on localhost:8000."
            st.session_state.page = "landing"
            st.rerun()
        except requests.exceptions.Timeout:
            st.session_state.last_error = "The analysis took too long and timed out. Try again with fewer pages or review the backend logs."
            st.session_state.page = "landing"
            st.rerun()
        except Exception as exc:
            st.session_state.last_error = f"Unexpected error while processing the video: {exc}"
            st.session_state.page = "landing"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_empty_results() -> None:
    st.markdown(
        '<div class="empty-note">'
        'The analysis endpoint finished, but the dashboard could not fetch fresh metrics '
        'from the API yet. Please refresh once or run the analysis again.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_results() -> None:
    kpis = fetch_kpis()
    topics = fetch_topics() or []
    comments_payload = fetch_comments(limit=200)

    if not kpis or kpis.get("total_comments", 0) == 0:
        render_empty_results()
        if st.button("Analyze another video"):
            st.session_state.page = "landing"
            st.rerun()
        return

    title = st.session_state.last_video_title or fetch_video_title(st.session_state.last_video_id)
    total_comments = kpis.get("total_comments", 0)
    summary = sentiment_summary_text(kpis)

    st.markdown(f'<div class="result-video-title">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-hero">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-big">{total_comments:,} Comments Analysed</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="result-summary">{summary}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    fig = build_sentiment_bar_chart(kpis)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">About the comments</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-mini"><div class="stat-mini-label">Positive</div>'
            f'<div class="stat-mini-value">{kpis.get("positive_count", 0):,}</div>'
            '<div class="stat-mini-copy">Comments with an overall positive tone.</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-mini"><div class="stat-mini-label">Neutral</div>'
            f'<div class="stat-mini-value">{kpis.get("neutral_count", 0):,}</div>'
            '<div class="stat-mini-copy">Comments that are descriptive or mixed in tone.</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-mini"><div class="stat-mini-label">Negative</div>'
            f'<div class="stat-mini-value">{kpis.get("negative_count", 0):,}</div>'
            '<div class="stat-mini-copy">Comments with critical or unfavorable sentiment.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 1.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">About the analysis</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.05, 1])
    with col_left:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        topic_fig = build_topics_chart(topics)
        if topic_fig is not None:
            st.plotly_chart(topic_fig, use_container_width=True)
        else:
            st.info("No topic-level records are available yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown('##### Reading notes')
        st.write(
            "The summary above is generated from the dominant sentiment class and the total "
            "number of processed comments. Topic counts come from the zero-shot classifier "
            "stored in the API database."
        )
        st.write(
            f"This run used up to {int(st.session_state.max_pages)} page(s) of comments and "
            f"the following topic labels: {', '.join(st.session_state.candidate_labels)}."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 1.25rem"></div>', unsafe_allow_html=True)

    with st.expander("Inspect processed comments", expanded=False):
        if comments_payload and comments_payload.get("items"):
            df = pd.DataFrame(comments_payload["items"])
            keep = [
                "author",
                "text",
                "like_count",
                "sentiment_label",
                "sentiment_score",
                "topic_label",
                "topic_score",
                "published_at",
            ]
            df = df[[column for column in keep if column in df.columns]].rename(
                columns={
                    "author": "Author",
                    "text": "Comment",
                    "like_count": "Likes",
                    "sentiment_label": "Sentiment",
                    "sentiment_score": "Sentiment score",
                    "topic_label": "Topic",
                    "topic_score": "Topic score",
                    "published_at": "Published at",
                }
            )
            for column in ["Sentiment score", "Topic score"]:
                if column in df.columns:
                    df[column] = df[column].round(3)
            st.dataframe(df, use_container_width=True, height=420, hide_index=True)
        else:
            st.info("No comments are available to display.")

    col_primary, col_secondary = st.columns([1.2, 1.2])
    with col_primary:
        if st.button("Analyze another video", use_container_width=True):
            st.cache_data.clear()
            st.session_state.page = "landing"
            st.session_state.video_url = ""
            st.session_state.last_result = None
            st.session_state.last_error = None
            st.rerun()
    with col_secondary:
        if st.button("Refresh analysis", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    page = st.session_state.page
    if page == "landing":
        render_landing()
    elif page == "processing":
        render_processing()
    elif page == "results":
        render_results()
    else:
        st.session_state.page = "landing"
        st.rerun()


if __name__ == "__main__":
    main()
