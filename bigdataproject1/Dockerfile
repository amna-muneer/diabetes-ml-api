##########################################
# STEP 1: Base image with Python
##########################################
FROM python:3.10-bullseye
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk-headless wget curl && \
    rm -rf /var/lib/apt/lists/*

##########################################
# STEP 3: Install Spark
##########################################
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3

RUN wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz && \
    tar -xzf spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz -C /opt/ && \
    rm spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz

ENV SPARK_HOME=/opt/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION
ENV PATH="$SPARK_HOME/bin:$PATH"

##########################################
# STEP 4: Set working directory
##########################################
WORKDIR /app

##########################################
# STEP 5: Install Python dependencies
##########################################
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

##########################################
# STEP 6: Copy application code
##########################################
COPY . .

##########################################
# STEP 7: Expose port & run
##########################################
EXPOSE 5000

CMD ["spark-submit", "app.py"]

