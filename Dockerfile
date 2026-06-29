FROM python:3.11-slim

LABEL maintainer="Innomatics Research Labs"
LABEL description="Eco-Logic: Thermal-Aware RL Optimizer"

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc default-libmysqlclient-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch==2.1.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/       ./src/
COPY dashboard/ ./dashboard/
COPY data/      ./data/
COPY config.yaml .

RUN mkdir -p checkpoints logs

ENTRYPOINT ["python", "src/main.py"]
CMD ["train", "--episodes", "200", "--agent", "qlearning"]
