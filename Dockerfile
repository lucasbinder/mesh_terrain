FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=80 \
    APP_RUNTIME_MODE=production \
    DASH_DEBUG=0 \
    GUNICORN_BIND=127.0.0.1:8050 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=300

WORKDIR ${APP_HOME}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        libexpat1 \
        libgomp1 \
        nginx \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    dash==4.1.0 \
    dask==2026.3.0 \
    Flask==3.1.3 \
    gunicorn==23.0.0 \
    numba==0.63.1 \
    numpy==2.3.5 \
    pandas==2.3.3 \
    pillow==12.2.0 \
    plotly==6.6.0 \
    pyomp==0.5.1 \
    rasterio==1.5.0 \
    requests==2.33.1 \
    scipy==1.17.1 \
    tqdm==4.67.3 \
    xarray==2025.1.1 \
    xarray-spatial==0.9.5

COPY app.py ./app.py
COPY assets ./assets
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/start.sh /usr/local/bin/start-app

RUN chmod +x /usr/local/bin/start-app \
    && mkdir -p /var/cache/nginx /var/lib/nginx /var/log/nginx /run/nginx

EXPOSE 80

CMD ["/usr/local/bin/start-app"]
