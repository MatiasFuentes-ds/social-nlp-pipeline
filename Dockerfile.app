# Dockerfile.app
# Contenedor para Streamlit (app/dashboard.py) — puerto 8501

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copiar el código del dashboard
COPY app/ ./app/

# 4. Copiar configuración de Streamlit (sin echo — evita errores de escape)
RUN mkdir -p /root/.streamlit
COPY streamlit_config/credentials.toml /root/.streamlit/credentials.toml
COPY streamlit_config/config.toml      /root/.streamlit/config.toml

# 5. Exponer puerto de Streamlit
EXPOSE 8501

# 6. Comando de arranque
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0"]
