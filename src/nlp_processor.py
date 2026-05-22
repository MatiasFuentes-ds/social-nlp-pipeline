"""
src/nlp_processor.py

Motor de inferencia y almacenamiento para un pipeline de NLP end-to-end
que analiza comentarios de YouTube usando modelos de Hugging Face.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import pipeline

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping for cardiffnlp/twitter-roberta-base-sentiment-latest
# ---------------------------------------------------------------------------
_SENTIMENT_LABEL_MAP: Dict[str, str] = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    # Some checkpoints already return human-readable labels; keep them as-is.
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
_DDL_VIDEOS = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    processed_at TEXT NOT NULL
);
"""

_DDL_COMMENTS = """
CREATE TABLE IF NOT EXISTS comments (
    comment_id   TEXT PRIMARY KEY,
    video_id     TEXT NOT NULL REFERENCES videos(video_id),
    author       TEXT,
    text         TEXT,
    like_count   INTEGER DEFAULT 0,
    published_at TEXT
);
"""

_DDL_ANALYSIS = """
CREATE TABLE IF NOT EXISTS analysis (
    comment_id      TEXT PRIMARY KEY REFERENCES comments(comment_id),
    sentiment_label TEXT,
    sentiment_score REAL,
    topic_label     TEXT,
    topic_score     REAL
);
"""


class NLPProcessor:
    """Motor de inferencia NLP y persistencia en SQLite para comentarios de YouTube.

    Attributes:
        db_path: Ruta al archivo de base de datos SQLite.
        batch_size: Tamaño de lote para la inferencia de los modelos.
        candidate_labels: Etiquetas de clasificación zero-shot.
        sentiment_model_id: Identificador del modelo de sentimiento en Hugging Face Hub.
        zeroshot_model_id: Identificador del modelo zero-shot en Hugging Face Hub.
        device: Dispositivo de cómputo seleccionado (cuda / mps / cpu).
    """

    DEFAULT_CANDIDATE_LABELS: List[str] = [
        "Music",
        "Controversy",
        "Fashion",
        "Politics",
        "Religion",
    ]

    SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ZEROSHOT_MODEL_ID = "facebook/bart-large-mnli"

    def __init__(
        self,
        db_path: str = "data/databaser.db",
        batch_size: int = 16,
        candidate_labels: Optional[List[str]] = None,
    ) -> None:
        """Inicializa la conexión a la base de datos, el esquema y los pipelines.

        Args:
            db_path: Ruta al archivo SQLite. El directorio padre se crea si no existe.
            batch_size: Número de textos procesados por lote en cada pipeline.
            candidate_labels: Etiquetas zero-shot personalizadas. Si es None se usan
                las de ``DEFAULT_CANDIDATE_LABELS``.
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.candidate_labels = candidate_labels or self.DEFAULT_CANDIDATE_LABELS

        self.device = self._resolve_device()
        logger.info("Dispositivo de cómputo seleccionado: %s", self.device)

        self._conn: sqlite3.Connection = self._init_db()
        logger.info("Base de datos lista en: %s", db_path)

        self._sentiment_pipe = self._load_sentiment_pipeline()
        self._zeroshot_pipe = self._load_zeroshot_pipeline()
        logger.info("Pipelines cargados correctamente.")

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> str:
        """Detecta el dispositivo de cómputo disponible en orden de preferencia.

        Returns:
            ``"cuda"`` si hay GPU NVIDIA, ``"mps"`` si hay Apple Silicon,
            o ``"cpu"`` en cualquier otro caso.
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> sqlite3.Connection:
        """Crea la conexión a SQLite e inicializa el esquema si no existe.

        Returns:
            Objeto de conexión SQLite con ``row_factory`` configurado.

        Raises:
            RuntimeError: Si el directorio padre no puede crearse.
        """
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        with conn:
            conn.execute(_DDL_VIDEOS)
            conn.execute(_DDL_COMMENTS)
            conn.execute(_DDL_ANALYSIS)
        return conn

    def clear_database(self) -> None:
        """Elimina todos los registros de las tablas y reinicia los contadores.

        Vacía las tablas ``analysis``, ``comments`` y ``videos`` en orden
        (hijos antes que padres) para respetar las claves foráneas.
        """
        logger.info("Iniciando purga de la base de datos...")
        try:
            with self._conn:
                self._conn.execute("PRAGMA foreign_keys=OFF;")
                self._conn.execute("DELETE FROM analysis;")
                self._conn.execute("DELETE FROM comments;")
                self._conn.execute("DELETE FROM videos;")
                self._conn.execute("PRAGMA foreign_keys=ON;")

            try:
                with self._conn:
                    self._conn.execute(
                        "DELETE FROM sqlite_sequence "
                        "WHERE name IN ('videos', 'comments', 'analysis');"
                    )
            except sqlite3.OperationalError:
                logger.debug("Tabla sqlite_sequence no encontrada, omitiendo reinicio de IDs.")

            logger.info("Base de datos limpiada con éxito. Lista para nueva ingesta.")

        except sqlite3.Error as exc:
            logger.error("Error crítico durante la purga de la base de datos: %s", exc)

    def _upsert_video(self, video_id: str) -> None:
        """Inserta o reemplaza un registro en la tabla ``videos``.

        Args:
            video_id: Identificador único del video de YouTube.
        """
        processed_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO videos (video_id, processed_at) VALUES (?, ?);",
                    (video_id, processed_at),
                )
        except sqlite3.Error as exc:
            logger.error("Error al insertar video '%s': %s", video_id, exc)

    def _upsert_comments(self, comments: List[Dict[str, Any]]) -> None:
        """Inserta o reemplaza comentarios en la tabla ``comments``.

        Args:
            comments: Lista de diccionarios con las claves ``comment_id``,
                ``video_id``, ``author``, ``text``, ``like_count`` y
                ``published_at``.
        """
        rows: List[Tuple] = [
            (
                c["comment_id"],
                c["video_id"],
                c.get("author"),
                c.get("text"),
                c.get("like_count", 0),
                c.get("published_at"),
            )
            for c in comments
        ]
        sql = (
            "INSERT OR REPLACE INTO comments "
            "(comment_id, video_id, author, text, like_count, published_at) "
            "VALUES (?, ?, ?, ?, ?, ?);"
        )
        try:
            with self._conn:
                self._conn.executemany(sql, rows)
            logger.info("Upsert de %d comentarios completado.", len(rows))
        except sqlite3.Error as exc:
            logger.error("Error al insertar comentarios en lote: %s", exc)

    def _upsert_analysis(self, results: List[Dict[str, Any]]) -> None:
        """Inserta o reemplaza resultados de análisis en la tabla ``analysis``.

        Args:
            results: Lista de diccionarios con las claves ``comment_id``,
                ``sentiment_label``, ``sentiment_score``, ``topic_label``
                y ``topic_score``.
        """
        rows: List[Tuple] = [
            (
                r["comment_id"],
                r["sentiment_label"],
                r["sentiment_score"],
                r["topic_label"],
                r["topic_score"],
            )
            for r in results
        ]
        sql = (
            "INSERT OR REPLACE INTO analysis "
            "(comment_id, sentiment_label, sentiment_score, topic_label, topic_score) "
            "VALUES (?, ?, ?, ?, ?);"
        )
        try:
            with self._conn:
                self._conn.executemany(sql, rows)
            logger.info("Upsert de %d registros de análisis completado.", len(rows))
        except sqlite3.Error as exc:
            logger.error("Error al insertar análisis en lote: %s", exc)

    # ------------------------------------------------------------------
    # Pipeline loaders
    # ------------------------------------------------------------------

    def _load_sentiment_pipeline(self):
        """Carga el pipeline de análisis de sentimiento desde Hugging Face Hub.

        Returns:
            Pipeline de ``text-classification`` configurado en el dispositivo activo.
        """
        logger.info("Cargando modelo de sentimiento: %s", self.SENTIMENT_MODEL_ID)
        return pipeline(
            "text-classification",
            model=self.SENTIMENT_MODEL_ID,
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=512,
        )

    def _load_zeroshot_pipeline(self):
        """Carga el pipeline de clasificación zero-shot desde Hugging Face Hub.

        Returns:
            Pipeline de ``zero-shot-classification`` configurado en el dispositivo activo.
        """
        logger.info("Cargando modelo zero-shot: %s", self.ZEROSHOT_MODEL_ID)
        return pipeline(
            "zero-shot-classification",
            model=self.ZEROSHOT_MODEL_ID,
            device=0 if self.device == "cuda" else -1,
        )

    # ------------------------------------------------------------------
    # Inference (batch-aware)
    # ------------------------------------------------------------------

    def _run_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Ejecuta análisis de sentimiento sobre una lista de textos en lotes.

        Args:
            texts: Textos a analizar. Los textos vacíos se reemplazan por
                un espacio para evitar errores del tokenizador.

        Returns:
            Lista de dicts con ``label`` (str legible) y ``score`` (float).
            Si un texto falla, devuelve ``{"label": "unknown", "score": 0.0}``.
        """
        sanitized = [t if t and t.strip() else " " for t in texts]
        results: List[Dict[str, Any]] = []
        try:
            raw_outputs = self._sentiment_pipe(sanitized, batch_size=self.batch_size)
            for output in raw_outputs:
                raw_label = output.get("label", "LABEL_1")
                label = _SENTIMENT_LABEL_MAP.get(raw_label, raw_label.lower())
                results.append({"label": label, "score": round(output.get("score", 0.0), 4)})
        except Exception as exc:
            logger.error("Error durante la inferencia de sentimiento: %s", exc)
            results = [{"label": "unknown", "score": 0.0}] * len(texts)
        return results

    def _run_zeroshot_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Ejecuta clasificación zero-shot sobre una lista de textos en lotes.

        Args:
            texts: Textos a clasificar. Los textos vacíos se reemplazan por
                un espacio para evitar errores del tokenizador.

        Returns:
            Lista de dicts con ``label`` (etiqueta de mayor score) y
            ``score`` (float). Si un texto falla, devuelve
            ``{"label": "unknown", "score": 0.0}``.
        """
        sanitized = [t if t and t.strip() else " " for t in texts]
        results: List[Dict[str, Any]] = []
        try:
            raw_outputs = self._zeroshot_pipe(
                sanitized,
                candidate_labels=self.candidate_labels,
                batch_size=self.batch_size,
            )
            # La API puede devolver un dict (un solo texto) o una lista.
            if isinstance(raw_outputs, dict):
                raw_outputs = [raw_outputs]
            for output in raw_outputs:
                top_label = output["labels"][0]
                top_score = round(output["scores"][0], 4)
                results.append({"label": top_label, "score": top_score})
        except Exception as exc:
            logger.error("Error durante la inferencia zero-shot: %s", exc)
            results = [{"label": "unknown", "score": 0.0}] * len(texts)
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_batch(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Procesa un lote de comentarios: inferencia + enriquecimiento con resultados.

        Args:
            comments: Lista de diccionarios de comentarios. Cada dict debe
                contener al menos ``comment_id`` y ``text``.

        Returns:
            La misma lista de diccionarios enriquecida con las claves
            ``sentiment_label``, ``sentiment_score``, ``topic_label``
            y ``topic_score``.
        """
        if not comments:
            logger.warning("process_batch recibió una lista vacía.")
            return []

        texts: List[str] = [c.get("text", "") for c in comments]
        logger.info("Iniciando inferencia sobre %d textos.", len(texts))

        sentiment_results = self._run_sentiment_batch(texts)
        zeroshot_results = self._run_zeroshot_batch(texts)

        enriched: List[Dict[str, Any]] = []
        for comment, sent, topic in zip(comments, sentiment_results, zeroshot_results):
            enriched.append(
                {
                    **comment,
                    "sentiment_label": sent["label"],
                    "sentiment_score": sent["score"],
                    "topic_label": topic["label"],
                    "topic_score": topic["score"],
                }
            )
        return enriched

    def run(self, comments: List[Dict[str, Any]], video_id: str) -> None:
        """Punto de entrada principal: procesa y persiste comentarios de un video.

        Realiza la inferencia en lotes y guarda los resultados en las tablas
        ``videos``, ``comments`` y ``analysis`` de la base de datos SQLite.

        Args:
            comments: Lista de diccionarios de comentarios. Cada dict debe
                contener al menos las claves ``comment_id``, ``text`` y
                opcionalmente ``author``, ``like_count`` y ``published_at``.
            video_id: Identificador único del video de YouTube al que pertenecen
                los comentarios.
        """
        logger.info(
            "Iniciando pipeline para video_id='%s' con %d comentarios.",
            video_id,
            len(comments),
        )

        # Inyectar video_id a cada comentario si no está presente
        for c in comments:
            c.setdefault("video_id", video_id)

        self._upsert_video(video_id)
        self._upsert_comments(comments)

        enriched = self.process_batch(comments)

        analysis_records = [
            {
                "comment_id": e["comment_id"],
                "sentiment_label": e["sentiment_label"],
                "sentiment_score": e["sentiment_score"],
                "topic_label": e["topic_label"],
                "topic_score": e["topic_score"],
            }
            for e in enriched
        ]
        self._upsert_analysis(analysis_records)

        logger.info(
            "Pipeline completado para video_id='%s'. %d registros persistidos.",
            video_id,
            len(analysis_records),
        )

    def close(self) -> None:
        """Cierra la conexión a la base de datos de forma segura."""
        if self._conn:
            self._conn.close()
            logger.info("Conexión a la base de datos cerrada.")

    def __enter__(self) -> "NLPProcessor":
        """Soporte para gestor de contexto (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cierra recursos al salir del bloque with."""
        self.close()

    
# ---------------------------------------------------------------------------
# Entry point — mock test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    MOCK_VIDEO_ID = "dQw4w9WgXcQ"

    MOCK_COMMENTS: List[Dict[str, Any]] = [
        {
            "comment_id": "c001",
            "author": "Alice",
            "text": "This song is absolutely incredible! I've been listening on repeat for days.",
            "like_count": 312,
            "published_at": "2024-01-15T10:23:00Z",
        },
        {
            "comment_id": "c002",
            "author": "Bob",
            "text": "I can't believe how controversial this artist has become lately.",
            "like_count": 89,
            "published_at": "2024-01-16T08:00:00Z",
        },
        {
            "comment_id": "c003",
            "author": "Carol",
            "text": "The Yeezy fashion line is overrated and way too expensive for what it is.",
            "like_count": 44,
            "published_at": "2024-01-17T15:10:00Z",
        },
        {
            "comment_id": "c004",
            "author": "Dave",
            "text": "His religious views are really interesting and thought-provoking.",
            "like_count": 201,
            "published_at": "2024-01-18T12:45:00Z",
        },
        {
            "comment_id": "c005",
            "author": "Eve",
            "text": "Politicians should stay out of the music industry entirely.",
            "like_count": 55,
            "published_at": "2024-01-19T09:30:00Z",
        },
        {
            "comment_id": "c006",
            "author": "Frank",
            "text": "",  # Empty text — edge case
            "like_count": 0,
            "published_at": "2024-01-20T11:00:00Z",
        },
    ]

    logger.info("=== Iniciando test con datos mock ===")

    with NLPProcessor(db_path="data/databaser.db", batch_size=4) as processor:
        # Demostración del flujo completo con purga previa
        processor.clear_database()
        processor.run(comments=MOCK_COMMENTS, video_id=MOCK_VIDEO_ID)

    # Verificación simple de los resultados guardados
    conn = sqlite3.connect("data/databaser.db")
    conn.row_factory = sqlite3.Row
    logger.info("--- Resultados en la base de datos ---")
    rows = conn.execute(
        """
        SELECT c.comment_id, c.author, a.sentiment_label, a.sentiment_score,
               a.topic_label, a.topic_score
        FROM comments c
        JOIN analysis a ON c.comment_id = a.comment_id
        WHERE c.video_id = ?
        ORDER BY c.comment_id;
        """,
        (MOCK_VIDEO_ID,),
    ).fetchall()
    for row in rows:
        logger.info(
            "[%s] %s → sentiment=%s(%.2f) | topic=%s(%.2f)",
            row["comment_id"],
            row["author"],
            row["sentiment_label"],
            row["sentiment_score"],
            row["topic_label"],
            row["topic_score"],
        )
    conn.close()
    logger.info("=== Test completado exitosamente ===")
