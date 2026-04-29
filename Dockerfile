FROM python:3.12-slim

# Imposta la cartella di lavoro
WORKDIR /app

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

# Comando di avvio per Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "general_main:app"]