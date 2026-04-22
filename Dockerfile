FROM eclipse-temurin:11-jdk-jammy

ENV SPARK_VERSION=3.3.2
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV SPARK_MASTER_HOST=spark-master
ENV SPARK_NO_DAEMONIZE=true


RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-setuptools \
    net-tools \
    netcat \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    pyarrow \
    findspark \
    jupyterlab \
    seaborn \
    matplotlib \
    polars


RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
 && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
 && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} \
 && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz


RUN wget -q https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.3_2.12/1.3.1/iceberg-spark-runtime-3.3_2.12-1.3.1.jar -P ${SPARK_HOME}/jars/


RUN mkdir -p /opt/spark/logs /notebooks /warehouse


COPY spark-defaults.conf ${SPARK_HOME}/conf/


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /notebooks

EXPOSE 7077 8080 4040 8888

ENTRYPOINT ["/entrypoint.sh"]