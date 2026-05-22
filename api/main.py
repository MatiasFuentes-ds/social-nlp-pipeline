"""
api/main.py

API RESTful construida con FastAPI para servir los datos procesados de NLP
desde la base de datos SQLite hacia un frontend en Streamlit, e integrar
la ejecución del pipeline de análisis directamente desde un endpoint REST.

Uso (desarrollo local):
    uvicorn api.main:app --reload --port 8000

Documentación interactiva:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""

import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Generator, List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Resolución de rutas — independiente del CWD
# ---------------------------------------------------------------------------
# api/main.py → parent = api/ → parent.parent = raíz del proyecto
_BASE_DIR: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = _BASE_DIR / "src" / "data" / "databaser.db"

# Agrega src/ al path para que los imports de YoutubeDataClient y NLPProcessor
# funcionen sin importar desde qué directorio se lanza uvicorn.
_SRC_DIR: Path = _BASE_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.append(str(_SRC_DIR))

from youtube_client import YoutubeDataClient  # noqa: E402
from nlp_processor import NLPProcessor        # noqa: E402

# ---------------------------------------------------------------------------
# Regex para extraer video_id de cualquier formato estándar de YouTube
# ---------------------------------------------------------------------------
# Cubre:
#   https://www.youtube.com/watch?v=VIDEO_ID
#   https://youtu.be/VIDEO_ID
#   https://www.youtube.com/embed/VIDEO_ID
#   https://www.youtube.com/shorts/VIDEO_ID
_YT_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    """Extrae el video_id (11 caracteres) de una URL estándar de YouTube.

    Args:
        url: URL completa del video de YouTube en cualquiera de sus formatos.

    Returns:
        String de 11 caracteres con el video_id, o ``None`` si la URL
        no coincide con ningún formato reconocido.
    """
    match = _YT_VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


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
    version="1.1.0",
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
# Pydantic schemas
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


class AnalyzeURLRequest(BaseModel):
    """Cuerpo de la petición para analizar un video de YouTube por URL.

    Attributes:
        url: URL completa del video (soporta youtube.com/watch, youtu.be,
            embed y shorts).
        max_pages: Número máximo de páginas de comentarios a extraer.
            Se limita a 3 por defecto para proteger la cuota de la API de
            YouTube durante la etapa de desarrollo.
    """

    url: str = Field(
        ...,
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        description="URL del video de YouTube a analizar.",
    )
    max_pages: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Páginas de comentarios a extraer (1-10). Default: 3.",
    )


class AnalyzeURLResponse(BaseModel):
    """Respuesta devuelta tras completar el análisis de un video."""

    status: str = Field(..., examples=["success"])
    message: str
    video_id: str
    comments_processed: int


# ---------------------------------------------------------------------------
# Endpoints GET (sin modificaciones respecto a la versión anterior)
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
# Endpoint POST /api/analyze/url — pipeline completo desde URL de YouTube
# ---------------------------------------------------------------------------

@app.post(
    "/api/analyze/url",
    response_model=AnalyzeURLResponse,
    summary="Analizar video de YouTube por URL",
    tags=["Pipeline"],
    status_code=200,
)
def analyze_url(request: AnalyzeURLRequest) -> AnalyzeURLResponse:
    """Ejecuta el pipeline NLP completo a partir de una URL de YouTube.

    Flujo interno:
        1. Extrae el ``video_id`` de la URL recibida.
        2. Purga la base de datos mediante ``NLPProcessor.clear_database()``.
        3. Extrae comentarios con ``YoutubeDataClient.get_comments()``.
        4. Ejecuta inferencia y persiste con ``NLPProcessor.run()``.

    Args:
        request: Cuerpo de la petición con ``url`` y ``max_pages``.

    Returns:
        AnalyzeURLResponse con el status, video_id y cantidad de comentarios
        procesados.

    Raises:
        HTTPException 400: Si la URL no corresponde a un video de YouTube válido.
        HTTPException 404: Si el video no tiene comentarios o el ID es inválido.
        HTTPException 500: Si ocurre un fallo en la API de YouTube o en el modelo.
    """
    # -- Paso 1: Extraer video_id ------------------------------------------------
    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No se pudo extraer un video_id válido de la URL: '{request.url}'. "
                "Formatos aceptados: youtube.com/watch?v=..., youtu.be/..., "
                "youtube.com/shorts/..., youtube.com/embed/..."
            ),
        )

    try:
        # -- Paso 2: Purgar base de datos ----------------------------------------
        with NLPProcessor(db_path=str(DB_PATH)) as processor:
            processor.clear_database()

            # -- Paso 3: Extraer comentarios -------------------------------------
            client = YoutubeDataClient()
            comments: List[Dict] = client.get_comments(
                video_id=video_id,
                max_pages=request.max_pages,
            )

            # -- Validación: comentarios vacíos ----------------------------------
            if not comments:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No se encontraron comentarios para video_id='{video_id}'. "
                        "Verifica que el video exista, sea público y tenga comentarios habilitados."
                    ),
                )

            # -- Paso 4: Inferencia NLP + persistencia ---------------------------
            processor.run(comments=comments, video_id=video_id)

    except HTTPException:
        # Re-lanzar HTTPExceptions sin envolverlas en un 500
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado durante el procesamiento del video '{video_id}': {exc}",
        ) from exc

    return AnalyzeURLResponse(
        status="success",
        message=f"Video '{video_id}' procesado correctamente.",
        video_id=video_id,
        comments_processed=len(comments),
    )


# ---------------------------------------------------------------------------
# Entry point — desarrollo local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
