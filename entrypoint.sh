#!/bin/bash

echo "Starting Spark entrypoint..."

# Create warehouse directory if it doesn't exist
mkdir -p /warehouse

if [ "$SPARK_MODE" == "master" ]; then
    echo "Starting Spark Master..."
    exec $SPARK_HOME/bin/spark-class org.apache.spark.deploy.master.Master
elif [ "$SPARK_MODE" == "worker" ]; then
    echo "Starting Spark Worker connecting to $SPARK_MASTER_URL..."
    exec $SPARK_HOME/bin/spark-class org.apache.spark.deploy.worker.Worker $SPARK_MASTER_URL
else
    echo "Unknown SPARK_MODE: $SPARK_MODE. Starting custom command..."
    exec "$@"
fi
