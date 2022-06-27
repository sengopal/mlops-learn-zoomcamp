#!/usr/bin/env python
# coding: utf-8

import pickle
import sys
from datetime import datetime

import pandas as pd
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

categorical = ['PUlocationID', 'DOlocationID']

def get_paths(run_date):
    year = run_date.year
    month = run_date.month
    print(f"running for file:{year}/{month}")

    input_file = f'data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'data/pred-{year:04d}-{month:02d}.parquet'

    return input_file, output_file

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

@task
def apply_model(input_file, output_file,run_date):
    logger = get_run_logger()
    year = run_date.year
    month = run_date.month

    logger.info(f'reading the data from {input_file}...')
    df = read_data(input_file)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f"mean predicted duration: {y_pred.mean()}")

    df['predicted_duration'] = y_pred
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = df[['predicted_duration', 'ride_id']]
    df_result.to_parquet(output_file,engine='pyarrow',compression=None,index=False)

    logger.info(f'saving the result to {output_file}...')
    return output_file

@flow
def ride_duration_prediction(run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    input_file, output_file = get_paths(run_date)

    apply_model(input_file=input_file, output_file=output_file, run_date=run_date)


def run():
    year = int(sys.argv[1])  # 2021
    month = int(sys.argv[2])  # 3
    # run_id = sys.argv[4] # 'e1efc53e9bd149078b0c12aeaa6365df'

    ride_duration_prediction(run_date=datetime(year=year, month=month, day=1))


if __name__ == '__main__':
    run()
