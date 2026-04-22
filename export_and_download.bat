@echo off
echo ====================================================
echo         CSV and Parquet Export Utility
echo ====================================================
echo.

echo Creating necessary directories...
mkdir data\export\csv 2>nul
mkdir data\export\parquet 2>nul
mkdir data\temp_csv 2>nul
mkdir downloads 2>nul

echo.
echo Step 1: Exporting data from Iceberg to CSV and Parquet formats...
echo.

docker exec spark-master mkdir -p /data/export/csv /data/export/parquet /data/temp_csv

docker cp export_formats.py spark-master:/notebooks/
docker exec -it spark-master /opt/spark/bin/spark-submit /notebooks/export_formats.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error during export. Please check if the Iceberg table exists.
    echo You may need to process your data first using large_file_processor.py
    echo.
    goto :EOF
)

echo.
echo Step 2: Downloading exported files to local machine...
echo.

python download_exports.py --output downloads

echo.
echo ====================================================
echo     PROCESS COMPLETE
echo ====================================================
echo.
echo If successful, your exported files should be in:
echo  - CSV: .\downloads\csv\
echo  - Parquet: .\downloads\parquet\
echo.
echo You can also access these files directly from:
echo  - CSV: .\data\export\csv\
echo  - Parquet: .\data\export\parquet\
echo. 