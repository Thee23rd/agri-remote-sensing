FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agri_rs agri_rs
COPY app app
COPY data data

EXPOSE 8080

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
