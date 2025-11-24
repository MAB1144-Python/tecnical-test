FROM python:3.11-slim

WORKDIR /app

# Avoid buffering for faster logs
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ ./app
# Copy TechnicalInterview folder (FAQ and related files)
COPY "TechnicalInterview" ./TechnicalInterview

COPY "source" ./source

# Expose port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
