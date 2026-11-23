FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir trajectory-sdk==0.10.25
COPY harness.py /app/harness.py
COPY tasks.json /app/tasks.json
