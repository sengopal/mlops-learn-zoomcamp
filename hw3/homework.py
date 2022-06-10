import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
import  numpy as np
from datetime import date, time, datetime

from dateutil.relativedelta import relativedelta
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
import pickle


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()

    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@task
def get_paths(data_date=None):
    logger = get_run_logger()
    if data_date is None:
        data_date = date.today().strftime("%Y-%m-%d")

    curr_date = datetime.strptime(data_date, "%Y-%m-%d")

    train_date = datetime.strftime((curr_date - relativedelta(months=2)), "%Y-%m")
    val_date = datetime.strftime((curr_date - relativedelta(months=1)), "%Y-%m")

    train_path = f"./data/fhv_tripdata_{train_date}.parquet"
    val_path = f"./data/fhv_tripdata_{val_date}.parquet"
    logger.info(train_path)
    logger.info(val_path)
    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    categorical = ['PUlocationID', 'DOlocationID']

    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    # Q1 : TypeError: cannot unpack non-iterable PrefectFuture object. Need to add result() here
    lr, dv = train_model(df_train_processed, categorical).result()

    with open(f'model-{date}.pkl', 'wb') as handle:
        pickle.dump(dv, handle, protocol=pickle.HIGHEST_PROTOCOL)

    run_model(df_val_processed, categorical, dv, lr)


# main(date="2021-08-15")
# print(get_paths(data_date="2021-08-15"))

# Q2: The valition MSE is:
#  The MSE of validation is: 11.63703262871038

# Q3: file size of the DictVectorizer that we trained when the date is 2021-08-15?
# 13,000 bytes


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

#Q4: What is the Cron expression to run a flow at 9 AM every 15th of the month?
# 9 AM every 15th of the month - https://crontab.guru/#00_9_15_*_* - 0 9 15 * * // "At 09:00 on day-of-month 15."

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)

# Q5: How many flow runs are scheduled by Prefect in advanced? (3)

# Q6: What is the command to view the available work-queues?
# prefect work-queue ls





