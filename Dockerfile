FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt . 2>/dev/null || true
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true
COPY . .
EXPOSE 8080
CMD ["python", "-m", "http.server", "8080"]
