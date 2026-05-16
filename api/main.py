"""
api/main.py

API RESTful construida con FastAPI para servir los datos procesados de NLP
desde la base de datos SQLite hacia un frontend en Streamlit.

Uso (desarrollo local):
    python api/main.py
    uvicorn api.main:app --reload --port 8000

Documentación interactiva disponible en:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import sqlite3
from pathlib import Path
from typing import Generator, List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Resolución de ruta a la base de datos (absoluta, independiente del CWD)
# ---------------------------------------------------------------------------
# api/main.py → parent = api/ → parent.parent = raíz del proyecto
_BASE_DIR: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = _BASE_DIR / "src" / "data" / "databaser.db"

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="YouTube NLP Analytics API",
    description=(
        "API RESTful que expone métricas analíticas de sentimiento y temáticas "
        "extraídas de comentarios de YouTube mediante modelos de Hugging Face. "
        "Diseñada para alimentar dashboards interactivos en Streamlit."
    ),
    version="1.0.0",
    contact={
        "name": "NLP Pipeline",
        "url": "https://github.com/MatiasFuentes-ds/social-nlp-pipeline",
    },
    license_info={"name": "MIT"},
)

# ---------------------------------------------------------------------------
# CORS — permite conexiones desde el frontend Streamlit (puerto distinto)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency: gestión del ciclo de vida de la conexión SQLite
# ---------------------------------------------------------------------------

def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Generador de dependencia que provee una conexión SQLite por request.

    Configura ``row_factory = sqlite3.Row`` para que los resultados sean
    accesibles como diccionarios y garantiza el cierre de la conexión al
    finalizar cada request mediante ``yield``.

    Yields:
        sqlite3.Connection: Conexión activa a la base de datos.

    Raises:
        HTTPException: 500 si el archivo de base de datos no existe.
    """
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Base de datos no encontrada en: {DB_PATH}. "
                   "Ejecuta el pipeline primero.",
        )
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pydantic schemas — modelos de respuesta para Swagger UI
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Respuesta del health check."""

    status: str = Field(..., examples=["ok"])
    message: str = Field(..., examples=["NLP API is running"])


class KPIResponse(BaseModel):
    """Métricas globales de sentimiento sobre todos los comentarios analizados."""

    total_comments: int = Field(..., description="Total de comentarios procesados.")
    positive_count: int = Field(..., description="Comentarios con sentimiento positivo.")
    negative_count: int = Field(..., description="Comentarios con sentimiento negativo.")
    neutral_count: int = Field(..., description="Comentarios con sentimiento neutro.")


class SentimentBreakdown(BaseModel):
    """Desglose de sentimientos para un tema específico."""

    positive: int = Field(default=0)
    negative: int = Field(default=0)
    neutral: int = Field(default=0)


class TopicItem(BaseModel):
    """Agrupación de comentarios por temática con desglose de sentimientos."""

    topic_label: str = Field(..., description="Nombre de la temática.")
    count: int = Field(..., description="Total de comentarios en esta temática.")
    sentiment_breakdown: SentimentBreakdown = Field(
        ..., description="Distribución de sentimientos dentro del tema."
    )


class CommentItem(BaseModel):
    """Comentario individual enriquecido con su análisis NLP."""

    comment_id: str
    video_id: str
    author: str | None = None
    text: str | None = None
    like_count: int = Field(default=0)
    published_at: str | None = None
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    topic_label: str | None = None
    topic_score: float | None = None


class PaginatedComments(BaseModel):
    """Respuesta paginada de comentarios."""

    total: int = Field(..., description="Total de registros disponibles.")
    limit: int
    offset: int
    items: List[CommentItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["Status"],
)
def health_check() -> HealthResponse:
    """Verifica que la API está operativa.

    Returns:
        HealthResponse: Status ``ok`` si el servicio está en línea.
    """
    return HealthResponse(status="ok", message="NLP API is running")


@app.get(
    "/api/kpis",
    response_model=KPIResponse,
    summary="KPIs globales de sentimiento",
    tags=["Analytics"],
)
def get_kpis(db: sqlite3.Connection = Depends(get_db)) -> KPIResponse:
    """Retorna los indicadores clave de sentimiento sobre todos los comentarios.

    Realiza un único ``GROUP BY`` en SQLite para calcular los conteos de cada
    categoría de sentimiento sin procesar datos en Python.

    Args:
        db: Conexión SQLite inyectada por ``get_db``.

    Returns:
        KPIResponse: Totales de comentarios por categoría de sentimiento.

    Raises:
        HTTPException: 500 si ocurre un error en la consulta SQL.
    """
    sql = """
        SELECT
            COUNT(*)                                          AS total_comments,
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count,
            SUM(CASE WHEN sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS neutral_count
        FROM analysis;
    """
    try:
        row = db.execute(sql).fetchone()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Error en consulta KPIs: {exc}") from exc

    return KPIResponse(
        total_comments=row["total_comments"] or 0,
        positive_count=row["positive_count"] or 0,
        negative_count=row["negative_count"] or 0,
        neutral_count=row["neutral_count"] or 0,
    )


@app.get(
    "/api/topics",
    response_model=List[TopicItem],
    summary="Distribución de temáticas con desglose de sentimiento",
    tags=["Analytics"],
)
def get_topics(db: sqlite3.Connection = Depends(get_db)) -> List[TopicItem]:
    """Agrupa los comentarios por temática e incluye el desglose de sentimientos.

    Usa un único ``GROUP BY topic_label`` con expresiones ``CASE WHEN`` para
    calcular los conteos de sentimiento por tema directamente en SQLite.

    Args:
        db: Conexión SQLite inyectada por ``get_db``.

    Returns:
        Lista de ``TopicItem`` ordenada de mayor a menor número de comentarios.

    Raises:
        HTTPException: 500 si ocurre un error en la consulta SQL.
    """
    sql = """
        SELECT
            topic_label,
            COUNT(*)                                                       AS count,
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative,
            SUM(CASE WHEN sentiment_label = 'neutral'  THEN 1 ELSE 0 END) AS neutral
        FROM analysis
        WHERE topic_label IS NOT NULL
        GROUP BY topic_label
        ORDER BY count DESC;
    """
    try:
        rows = db.execute(sql).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Error en consulta Topics: {exc}") from exc

    return [
        TopicItem(
            topic_label=row["topic_label"],
            count=row["count"],
            sentiment_breakdown=SentimentBreakdown(
                positive=row["positive"] or 0,
                negative=row["negative"] or 0,
                neutral=row["neutral"] or 0,
            ),
        )
        for row in rows
    ]


@app.get(
    "/api/comments",
    response_model=PaginatedComments,
    summary="Listado paginado de comentarios con análisis NLP",
    tags=["Comments"],
)
def get_comments(
    limit: int = 50,
    offset: int = 0,
    db: sqlite3.Connection = Depends(get_db),
) -> PaginatedComments:
    """Retorna una lista paginada de comentarios con sus resultados de análisis.

    Une las tablas ``comments`` y ``analysis`` para entregar datos enriquecidos.
    La paginación se delega a SQLite mediante ``LIMIT`` y ``OFFSET``.

    Args:
        limit: Número máximo de comentarios a devolver (default=50).
        offset: Número de registros a saltar para la paginación (default=0).
        db: Conexión SQLite inyectada por ``get_db``.

    Returns:
        PaginatedComments con el total de registros y la página solicitada.

    Raises:
        HTTPException: 500 si ocurre un error en la consulta SQL.
    """
    count_sql = "SELECT COUNT(*) AS total FROM comments;"
    data_sql = """
        SELECT
            c.comment_id,
            c.video_id,
            c.author,
            c.text,
            c.like_count,
            c.published_at,
            a.sentiment_label,
            a.sentiment_score,
            a.topic_label,
            a.topic_score
        FROM comments c
        LEFT JOIN analysis a ON c.comment_id = a.comment_id
        ORDER BY c.published_at DESC
        LIMIT ? OFFSET ?;
    """
    try:
        total: int = db.execute(count_sql).fetchone()["total"]
        rows = db.execute(data_sql, (limit, offset)).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500, detail=f"Error en consulta Comments: {exc}"
        ) from exc

    items = [CommentItem(**dict(row)) for row in rows]

    return PaginatedComments(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )


# ---------------------------------------------------------------------------
# Entry point — desarrollo local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

