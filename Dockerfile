# One image, one service: API + client + audio work.
FROM python:3.12-slim

# ffmpeg/ffprobe are not optional - the app refuses to pretend it can
# find audio without them, and /api/health says so.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server
COPY web ./web
COPY tests ./tests

# Railway (and most hosts) mount a volume; point QC_DATA_DIR at it so the
# sqlite file and the saved clips survive a redeploy.
ENV QC_DATA_DIR=/data \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]

EXPOSE 8080
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
