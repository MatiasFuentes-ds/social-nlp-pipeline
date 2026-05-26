# Social Media NLP Pipeline

Un pipeline de datos *end-to-end* diseñado para la extracción, procesamiento de lenguaje natural (NLP) y visualización interactiva de comentarios masivos en redes sociales.

Este proyecto implementa una arquitectura orientada a microservicios, desacoplando la ingesta de datos, el motor de inferencia y la interfaz de usuario, garantizando escalabilidad y un despliegue ágil mediante contenedores.

***

## Arquitectura y Tecnologías

- **Ingesta de Datos:** Extracción automatizada y paginada mediante **YouTube Data API v3**.
- **Procesamiento NLP:** Modelos pre-entrenados de **Hugging Face** ejecutados localmente por lotes (*batching*).
  - *Análisis de Sentimiento:* `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - *Clasificación de Temáticas (Zero-Shot):* `cross-encoder/nli-MiniLM2-L6-H768`
  - Los *candidate labels* son configurables desde la interfaz sin modificar el código.
- **Almacenamiento:** Base de datos relacional **SQLite** con lógica estricta de *upsert* y purga de estado para garantizar datos siempre frescos.
- **Backend REST API:** Construido con **FastAPI** para servir los datos procesados de forma asíncrona y robusta.
- **Frontend Interactivo:** Interfaz de usuario Single Page Application (SPA) desarrollada puramente en Python con **Streamlit** y visualizaciones de **Plotly**.
- **Despliegue:** Contenerización de servicios mediante **Docker** y orquestación con **Docker Compose**.

***

## Requisitos Previos

Para ejecutar este proyecto de forma local, necesitas tener instalado:

- [Docker](https://docs.docker.com/get-docker/) y Docker Compose.
- Una clave válida de la API de YouTube (Google Cloud Console).

***

## Configuración del Entorno (Archivo `.env`)

Por motivos de seguridad, las credenciales no están incluidas en el repositorio. Para que los contenedores de Docker puedan acceder a la API de YouTube durante la ejecución, debes crear un archivo de variables de entorno.

1. Crea un archivo llamado exactamente `.env` en el directorio raíz de este proyecto (al mismo nivel que el archivo `docker-compose.yml`).
2. Abre el archivo e inserta tu clave de API con el siguiente formato:

```env
YOUTUBE_API_KEY=TuClaveDeApiAqui
```

> **Nota:** Docker Compose está configurado para leer automáticamente este archivo `.env` en la raíz e inyectar las variables en el contenedor del backend.

***

## Cómo Ejecutar el Proyecto

El proyecto está diseñado para levantarse con un solo comando. Abre tu terminal en la raíz del proyecto y ejecuta:

```bash
docker compose up --build
```

Docker se encargará de descargar las imágenes de Python, instalar las dependencias aisladas, configurar la red interna y levantar ambos servicios.

### Acceso a los Servicios

Una vez que los contenedores estén corriendo, puedes acceder a las herramientas a través de tu navegador:

| Servicio | URL |
|---|---|
| Dashboard Interactivo (Streamlit) | `http://localhost:8501` |
| Documentación de la API (Swagger UI) | `http://localhost:8000/docs` |

***

## Estructura del Repositorio

```text
social-nlp-pipeline/
├── api/                  # Código fuente de la API RESTful (FastAPI)
├── app/                  # Código fuente del frontend (Streamlit)
├── src/                  # Módulos core: extracción desde YouTube y procesamiento NLP
├── streamlit_config/     # Archivos de configuración de Streamlit para Docker
├── .gitignore
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
├── .dockerignore
└── requirements.txt
```

***

*Desarrollado por [Matías Fuentes](https://github.com/MatiasFuentes-ds) como proyecto de portafolio en Ingeniería de Datos end-to-end.*
