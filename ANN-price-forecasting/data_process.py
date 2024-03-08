## Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

import os
import json

EPEX_DATA_DIR_PATH = "..\data\EPEX_spot"
CSV_FILE_PATH = f"{EPEX_DATA_DIR_PATH}\EPEX-spot.csv"

EMP_DATA_DIR_PATH = "..\data\energymarketprice"
RTE_DATA_DIR_PATH = "..\data\RTE_demand"

## Functions to process the input

def csv_to_dataframe(csv_path = f"{EMP_DATA_DIR_PATH}/FR_pwr_spot_base_5years.csv"):
    '''
    Get the data from a csv file, process it and returns the corresponding pandas.dataframe
    '''
    df = pd.read_csv(csv_path, sep=";")

    # Convert date column to Date type
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Convert value column to float type
    df['Value'] = df['Value'].str.replace(',', '.').astype(float)

    return df

# This function process the data found on :
# https://www.services-rte.com/fr/telechargez-les-donnees-publiees-par-rte.html?category=consumption&type=weekly_forecasts&subType=D-2
def get_consumption_data(file_paths= {2019: f"{RTE_DATA_DIR_PATH}/prevision_conso_2019.csv", 2020: f"{RTE_DATA_DIR_PATH}/prevision_conso_2020.csv", 2021: f"{RTE_DATA_DIR_PATH}/prevision_conso_2021.csv", 2022: f"{RTE_DATA_DIR_PATH}/prevision_conso_2022.csv", 2023: f"{RTE_DATA_DIR_PATH}/prevision_conso_2023.csv"}):        
    '''
    Read the data in every file and add it to a dataframe.

    Returns: a dataframe of the consumption of each year with four columns:
        - Date: the date
        - tier1: the mean consumption during the first tier of the day
        - tier2: the mean consumption during the second tier of the day
        - tier3: the mean consumption during the third tier of the day 
    '''
    
    consumption = pd.DataFrame()

    for year in file_paths:
        df = pd.read_csv(file_paths[year], sep=";")
        day_column = df["Jours/Heures"]

        df = df.drop(columns=['Jours/Heures'])

        # Compute the mean of foreseen consumption during each tier
        # and store the result in a new df
        l = len(df.columns) // 3
        means_df = pd.DataFrame()

        #means_df['Forecasted demand'] = df.mean(axis=1)/1000
        grouped_columns = [df.columns[:l], df.columns[l:2*l], df.columns[2*l:]]
        means_df['Forecasted demand1'] = df[grouped_columns[0]].mean(axis=1)
        means_df['Forecasted demand2'] = df[grouped_columns[1]].mean(axis=1)
        means_df['Forecasted demand3'] = df[grouped_columns[2]].mean(axis=1)
        means_df.insert(0, 'Date', day_column)

        # Concat the dataframe to the final result
        consumption = pd.concat([consumption, means_df], ignore_index=True)
                
    consumption['Date'] = pd.to_datetime(consumption['Date'], format=' %d/%m/%Y')

    consumption['Change in demand1'] = consumption['Forecasted demand1'].diff()
    consumption['Change in demand2'] = consumption['Forecasted demand2'].diff()
    consumption['Change in demand3'] = consumption['Forecasted demand3'].diff()
    consumption.iloc[1:]
    return consumption
   

def lineplot_from_dataframe(df, x, y, title):
    '''
    Create a Seaborn line plot from a Pandas DataFrame.

    Parameters:
    - df: the dataframe
    - x: The column from the DataFrame that will be on the x-axis
    - y: The column from the DataFrame that will be on the y-axis
    - title: the title of the plot
    '''
    plt.figure(figsize=(14,4))
    sns.lineplot(x=x, y=y, data=df)
    plt.title(title)
    plt.show()


def prepare_df_for_price_forecasting(df):
    '''
    The modifications are:
    - Remove Unit column
    - Add:
        - value_one_day_before
        - value_one_week_before
        - value_two_weeks_before
        - value_three_weeks_before
        - value_four_weeks_before
        - value_to_predict (next day)
        - day of the week (monday = 0, ..., sunday = 6)
    - Remove the first 28 columns (to avoid NaN values)
    - Keep only the values from 2019 to 2021 where the price was stable
    '''

    # Supprimer la première et la dernière colonne
    nn_df = df.drop(['Currency/Unit'], axis=1)

    # Supprimer la colonne 'Date'
    nn_df['Date'] = pd.to_datetime(df['Date'])

    # Add a column for the day of the week
    nn_df['Day of Week'] = nn_df['Date'].apply(lambda x: datetime.weekday(x) + 1)
    nn_df = nn_df.drop(['Date'], axis=1)

    # Ajouter les colonnes demandées
    nn_df['value_two_days_before'] = nn_df['Value'].shift(1)

    nn_df['value_one_week_before1'] = nn_df['Value'].shift(7)
    nn_df['value_one_week_before2'] = nn_df['Value'].shift(8)

    nn_df['value_two_weeks_before1'] = nn_df['Value'].shift(14)
    nn_df['value_two_weeks_before2'] = nn_df['Value'].shift(15)

    nn_df['value_three_weeks_before1'] = nn_df['Value'].shift(21)
    nn_df['value_three_weeks_before2'] = nn_df['Value'].shift(22)    

    nn_df['value_four_weeks_before1'] = nn_df['Value'].shift(28)
    nn_df['value_four_weeks_before2'] = nn_df['Value'].shift(29)

    # Value to predict
    nn_df['value_to_predict'] = nn_df['Value'].shift(-1)

    # Supprimer les premières lignes pour ne pas avoir de NaN
    nn_df = nn_df.iloc[30:-30]

    # Réinitialiser l'index
    nn_df = nn_df.reset_index(drop=True)

    return nn_df


def merge_dataframes(df1, df2):
    return pd.merge(df1, df2, on='Date')

def prepare_datasets_for_model(X, y, train_batch_size, test_batch_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_data = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float), y_train)
    test_data = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float), y_test)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    test_loader = DataLoader(test_data, batch_size=test_batch_size)
    
    num_inputs = X_train.shape[1]

    return train_loader, test_loader, y_test, num_inputs


## Process the output
def get_stats_from_result(prediction_error):
    error_dic = {
        '99%': 0,
        '95%': 0,
        '90%': 0,
        '75%': 0,
        '50%': 0,
    }

    positive_error_dic = {
        '99%': 0,
        '95%': 0,
        '90%': 0,
        '75%': 0,
        '50%': 0,
    }

    percent_value = 100 / len(prediction_error)

    for error in prediction_error:
        is_negative = error <= 0
        error = np.abs(error)

        if error < 1:
            error_dic['99%'] += percent_value        
        if error < 5:
            error_dic['95%'] += percent_value
        if error < 10:
            error_dic['90%'] += percent_value
        if error < 25:
            error_dic['75%'] += percent_value
        if error < 50:
            error_dic['50%'] += percent_value
        
        if error < 1 or is_negative:
            positive_error_dic['99%'] += percent_value        
        if error < 5 or is_negative:
            positive_error_dic['95%'] += percent_value
        if error < 10 or is_negative:
            positive_error_dic['90%'] += percent_value
        if error < 25 or is_negative:
            positive_error_dic['75%'] += percent_value
        if error < 50 or is_negative:
            positive_error_dic['50%'] += percent_value
        
    return ( pd.DataFrame(error_dic, index=[0]), pd.DataFrame(positive_error_dic, index=[0]) )

def write_csv(number_of_years, force=False):

    if force or not os.path.exists(CSV_FILE_PATH):
        years = ["2021", "2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013", "2012", "2011"]
        data = []

        for year in years[:number_of_years]:
            f = open(f'{EPEX_DATA_DIR_PATH}/EPEX-spot-{year}.json')
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


def epex_csv_to_dataframe():

    assert os.path.exists(CSV_FILE_PATH), "The data was not generated yet, you can generate it with the write_csv function"

    df = pd.read_csv( CSV_FILE_PATH , sep=",")

    # Convert date column to Date type
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Convert value column to float type
    df['price_euros_mwh'] = df['price_euros_mwh'].astype(float)

    return df