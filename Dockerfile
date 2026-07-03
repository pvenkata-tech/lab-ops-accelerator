FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lab_ops_accelerator/ ./lab_ops_accelerator/
COPY samples/ ./samples/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "lab_ops_accelerator.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
