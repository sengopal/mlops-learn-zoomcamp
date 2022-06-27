FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install scikit-learn==1.0.2 flask pandas mlflow prefect==2.0b5 pyarrow

#COPY ["Pipfile","Pipfile.lock", "./"]

COPY ["*.py", "./"]

RUN mkdir -p /app/data

COPY ["data/*.parquet","./data"]

WORKDIR /app

CMD ["python", "./starter.py", "2021", "4"]
