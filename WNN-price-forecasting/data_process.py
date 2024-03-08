## Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json

DATA_DIR_PATH = "..\data\EPEX_spot"
CSV_FILE_PATH = f"{DATA_DIR_PATH}\EPEX-spot.csv"

## Functions to process the input

def write_csv(number_of_years, force=False):

    if force or not os.path.exists(CSV_FILE_PATH):
        years = ["2021", "2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013", "2012", "2011"]
        data = []

        for year in years[:number_of_years]:
            f = open(f'{DATA_DIR_PATH}/EPEX-spot-{year}.json')
            data += json.load(f)
        
        df = pd.DataFrame(data)

        df['startDate'] = pd.to_datetime(df['startDate'], format='%Y/%m/%d %H', utc=True)
        df['date'] = df['startDate'].dt.date
        df['hour'] = df['startDate'].dt.hour
        df['day_of_week'] = df['startDate'].dt.dayofweek    

        columns_to_drop = ['startDate', 'endDate']    
        if 'volume_mwh' in df:
            columns_to_drop.append( 'volume_mwh' )        

        df = df.drop(columns_to_drop, axis=1)        

        columns_order = ["date", "hour", "day_of_week", "price_euros_mwh"]
        df = df[columns_order]

        df.to_csv(CSV_FILE_PATH, index=False)


def csv_to_dataframe():

    assert os.path.exists(CSV_FILE_PATH), "The data was not yet generated, you can generate it with the write_csv function"

    df = pd.read_csv( CSV_FILE_PATH , sep=",")

    # Convert date column to Date type
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Convert value column to float type
    df['price_euros_mwh'] = df['price_euros_mwh'].astype(float)

    return df