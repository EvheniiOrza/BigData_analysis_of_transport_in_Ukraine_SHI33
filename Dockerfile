# ===== Base image: Java 8
FROM eclipse-temurin:8-jdk

# ===== Змінні для версій
ARG PYSPARK_VERSION=3.2.0
ENV VENV_PATH=/opt/venv

# ===== Встановлюємо Python 3 та pip
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip curl wget unzip \
 && rm -rf /var/lib/apt/lists/*

# ===== Створюємо virtual environment для Python
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# ===== Встановлюємо PySpark
RUN pip install --no-cache-dir --default-timeout=300 pyspark==$PYSPARK_VERSION

# ===== Робоча директорія проєкту
WORKDIR /app
COPY . .

# ===== Точка входу
CMD ["python", "main.py"]