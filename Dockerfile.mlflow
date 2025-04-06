FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip \
    mlflow psycopg2-binary

EXPOSE 5005

CMD ["mlflow", "server", \
    "--backend-store-uri", "postgresql://mlflow_user:mlflow_password@postgres:5432/mlflow_db", \
    "--default-artifact-root", "/mlflow/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5005"]
