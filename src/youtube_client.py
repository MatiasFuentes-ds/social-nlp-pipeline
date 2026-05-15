# Archivo dedicado a conectarse a la API de Youtube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s %(message)s'
)

class YoutubeDataClient:
    def __init__(self):
        """ Inicializa la conexión con la API de Youtube"""
        load_dotenv()
        api_key = os.getenv("YOUTUBE_API_KEY")

        logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

        if not api_key:
            raise ValueError("API_KEY NO ENCONTRADA -- REVISAR .ENV")

        self.youtube = build('youtube','v3',developerKey=api_key)

    def get_comments(self, video_id:str, max_pages:int=5) -> list:
        comments_data = []
        next_page_token = None
        pages_fetched = 0

        logging.info(f"EXTRACCIÓN INICIADA -- ID DE VIDEO: {video_id}")
    
        try:
            while pages_fetched < max_pages:
                request = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=100,
                        pageToken=next_page_token,
                        textFormat="plainText",
                )
                response = request.execute()

                for item in response.get('items',[]):
                    comment = item['snippet']['topLevelComment']['snippet']

                    # Estructuramos sólo la data util.
                    comments_data.append({
                        'comment_id': item['id'],
                        'author': comment.get('authorDisplayName', 'Unknown'),
                        'text': self._clean_text(comment.get('textDisplay', '')),
                        'like_count': comment.get('likeCount', 0),
                        'published_at': comment.get('publishedAt')
                    })

                next_page_token = response.get('nextPageToken')
                pages_fetched += 1
                logging.info(f"Página {pages_fetched} extraída. Total acumulado: {len(comments_data)}")

                # Si no hay más páginas, rompemos el bucle
                if not next_page_token:
                    logging.info("No hay más páginas disponibles.")
                    break
        except HttpError as e:
            logging.error(f"Error HTTP al conectar con la API de YouTube: {e}")
        except Exception as e:
            logging.error(f"Error inesperado: {e}")

        logging.info(f"Extracción finalizada: {len(comments_data)} comentarios en total.")

        return comments_data

    def _clean_text(self, text: str) -> str:
        """
        Limpia el texto base para facilitar el trabajo del modelo NLP.
        """
        if not text:
            return ""
        # Quitamos saltos de línea y tabulaciones que rompen los CSV/DataFrames
        text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        # Quitamos espacios dobles
        return ' '.join(text.split())


# ==========================================
# Bloque de prueba.
# ==========================================
if __name__ == "__main__":
    # Instanciamos el cliente
    client = YoutubeDataClient()
    
    # Probamos con un ID de video cualquiera.
    test_video_id = "XS8-zm7Cmho" 
    
    # Extraemos solo 2 páginas (aprox 200 comentarios) para no gastar cuota de la API
    extracted_data = client.get_comments(video_id=test_video_id, max_pages=2)
    
    if extracted_data:
        print("\n--- MUESTRA DEL PRIMER COMENTARIO ---")
        print(f"Autor: {extracted_data[0]['author']}")
        print(f"Texto: {extracted_data[0]['text']}")
        print(f"Likes: {extracted_data[0]['like_count']}")
        print(f"Fecha: {extracted_data[0]['published_at']}")
