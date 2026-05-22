"""
src/run_pipeline.py

Orquestador principal (entrypoint) del pipeline ETL/Inferencia para análisis
de comentarios de YouTube. Une la extracción de datos (YoutubeDataClient)
con el procesamiento NLP y almacenamiento (NLPProcessor).

Uso:
    python src/run_pipeline.py --video_id <ID> [--max_pages N] [--batch_size N]

Ejemplo:
    python src/run_pipeline.py --video_id dQw4w9WgXcQ --max_pages 3 --batch_size 32
"""

import argparse
import logging
import sys
import time
from typing import List, Dict, Any

from youtube_client import YoutubeDataClient
from nlp_processor import NLPProcessor

# ---------------------------------------------------------------------------
# Logging — configuración centralizada del módulo
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Construye y devuelve el parser de argumentos de línea de comandos.

    Returns:
        argparse.ArgumentParser configurado con todos los argumentos del pipeline.
    """
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "Pipeline ETL/Inferencia: extrae comentarios de un video de YouTube "
            "y aplica análisis de sentimiento y clasificación zero-shot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_id",
        type=str,
        required=True,
        metavar="VIDEO_ID",
        help="ID del video de YouTube a analizar (ej. dQw4w9WgXcQ).",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=5,
        metavar="N",
        help="Número máximo de páginas de comentarios a extraer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="Tamaño del lote para la inferencia NLP.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def extract_comments(
    video_id: str,
    max_pages: int,
) -> List[Dict[str, Any]]:
    """Paso 1 del pipeline: extrae comentarios desde YouTube.

    Args:
        video_id: Identificador único del video de YouTube.
        max_pages: Número máximo de páginas de resultados a recuperar.

    Returns:
        Lista de diccionarios con los datos de cada comentario.
        Devuelve una lista vacía si no se encuentran comentarios.
    """
    logger.info(
        "Iniciando extracción | video_id='%s' | max_pages=%d",
        video_id,
        max_pages,
    )
    client = YoutubeDataClient()
    comments: List[Dict[str, Any]] = client.get_comments(
        video_id=video_id,
        max_pages=max_pages,
    )
    logger.info(
        "Extracción completada | %d comentarios obtenidos para video_id='%s'.",
        len(comments),
        video_id,
    )
    return comments


def run_nlp(
    comments: List[Dict[str, Any]],
    video_id: str,
    batch_size: int,
) -> None:
    """Paso 2 del pipeline: procesa los comentarios con NLP y persiste en SQLite.

    Args:
        comments: Lista de diccionarios de comentarios extraídos.
        video_id: Identificador único del video de YouTube.
        batch_size: Número de textos procesados simultáneamente por el pipeline.
    """
    logger.info(
        "Iniciando procesamiento NLP | video_id='%s' | batch_size=%d",
        video_id,
        batch_size,
    )
    with NLPProcessor(batch_size=batch_size) as processor:
        processor.clear_database() # purgar datos anteriores pre nueva ingesta
        processor.run(comments, video_id) # procesar nuevo video :)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Función principal que orquesta el pipeline completo ETL/Inferencia.

    Flujo:
        1. Parsea los argumentos de línea de comandos.
        2. Inicia el temporizador total del pipeline.
        3. Extrae comentarios del video de YouTube indicado.
        4. Valida que existan comentarios; termina limpiamente si no hay.
        5. Ejecuta el procesamiento NLP y persiste en SQLite.
        6. Registra el tiempo total de ejecución y el mensaje de éxito.

    Raises:
        SystemExit: Con código 1 si ocurre cualquier error inesperado,
            o con código 0 si no hay comentarios que procesar.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    logger.info(
        "Pipeline iniciado | video_id='%s' | max_pages=%d | batch_size=%d",
        args.video_id,
        args.max_pages,
        args.batch_size,
    )

    # Temporizador global del pipeline
    pipeline_start: float = time.time()

    try:
        # -- Paso 3: Extracción ---------------------------------------------------
        comments = extract_comments(
            video_id=args.video_id,
            max_pages=args.max_pages,
        )

        # Validación: lista vacía → warning + salida limpia
        if not comments:
            logger.warning(
                "No se encontraron comentarios para video_id='%s'. "
                "El pipeline finaliza sin procesar datos.",
                args.video_id,
            )
            sys.exit(0)

        # -- Paso 4: Procesamiento NLP + persistencia ----------------------------
        run_nlp(
            comments=comments,
            video_id=args.video_id,
            batch_size=args.batch_size,
        )

        # -- Paso 5: Métricas de ejecución ---------------------------------------
        elapsed: float = time.time() - pipeline_start
        logger.info(
            "Pipeline completado exitosamente | video_id='%s' | "
            "comentarios procesados=%d | tiempo total=%.2fs",
            args.video_id,
            len(comments),
            elapsed,
        )

    except KeyboardInterrupt:
        elapsed = time.time() - pipeline_start
        logger.warning(
            "Pipeline interrumpido por el usuario (KeyboardInterrupt) "
            "tras %.2fs de ejecución.",
            elapsed,
        )
        sys.exit(1)

    except MemoryError:
        elapsed = time.time() - pipeline_start
        logger.error(
            "Error de memoria insuficiente tras %.2fs. "
            "Considera reducir --max_pages o --batch_size.",
            elapsed,
        )
        sys.exit(1)

    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - pipeline_start
        logger.error(
            "Error inesperado tras %.2fs de ejecución: %s",
            elapsed,
            exc,
            exc_info=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

