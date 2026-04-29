FROM python:3.12-slim

# Imposta la cartella di lavoro
WORKDIR /app

# Variabili d'ambiente per evitare problemi con i log e i file .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installa le dipendenze di sistema necessarie per OpenCV e OnnxRuntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia e installa i requisiti Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice sorgente
COPY . .

# FONDAMENTALE: Espone la porta per Azure
EXPOSE 8000

# Comando di avvio per Gunicorn
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "general_main:app"]
