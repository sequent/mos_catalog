# Поиск аналогов: один образ, одна команда (docker compose up). Модель, каталог и индекс строятся при сборке,
# поэтому контейнер стартует за секунды и работает без интернета.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/pipeline \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# 1. зависимости (слой меняется только вместе с requirements-runtime.txt)
COPY requirements-runtime.txt constraints.txt ./
RUN pip install --no-cache-dir -c constraints.txt -r requirements-runtime.txt

# 2. модель эмбеддингов (≈470 МБ) внутри образа; слой кэшируется, пока не изменён скрипт загрузки
COPY scripts/download_model.py scripts/download_model.py
RUN python -I scripts/download_model.py

# 3. обработка базы и эмбеддинги: зависят от кода, словаря и xlsx, но не от правил
COPY pipeline pipeline
COPY scripts scripts
COPY config/dictionaries.yaml config/dictionaries.yaml
ENV CATALOG_XLSX=/app/source/catalog.xlsx
COPY *.xlsx source/catalog.xlsx
RUN python -I scripts/prepare_catalog.py && python -I scripts/build_index.py

# 4. правила и страница: меняются чаще всего, пересборка занимает секунды
COPY config/rules.yaml config/assumptions.yaml config/
COPY web web

RUN useradd --system --no-create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "--app-dir", "pipeline", "analog_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
