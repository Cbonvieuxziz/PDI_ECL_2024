# Importation des bibliothèques nécessaires
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Import of root_mean_squared_error from sklearn.metrics failed
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt( mean_squared_error(y_true, y_pred) )

class WNN_electricity_price_predictor:

    def __init__(self):
        # weights='distance' for inverse distance weighting    
        self.model = KNeighborsRegressor(n_neighbors=2, weights='distance')

    def set_train_test_sets(self, df : pd.DataFrame, prediction_horizon : int, num_input_days : int = 200):
        """ 
        Returns the train and test datasets from a dataframe

        Params:
        - df: the dataframe constructed by csv_to_dataframe
        - prediction_horizon: The time interval for which predictions are to be made. 
        For example, if prediction_interval is set to 1, the function predicts for the next day.
        If set to 7, the function predicts for the next week, and so on.
        - num_input_days: The number of past days to be considered as input for the machine learning model
        """

        number_of_values = df.shape[0]

        sample_size = num_input_days * 24
        prediction_horizon *= 24

        X, y = [], []
        raw_values = df[['hour', 'day_of_week', 'price_euros_mwh']].values
        
        for i in range( number_of_values - sample_size - prediction_horizon ):
            X.append( raw_values[i:i + sample_size] )
            y.append( raw_values[i + sample_size + prediction_horizon][2] )

        # Convert X to a 2d array
        X = np.array(X)
        nsamples, nx, ny = X.shape    
        X = X.reshape((nsamples, nx*ny))

        # Separate train test sets and return them
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):  
        """
        Train the model
        """                        
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, num_data_to_plot=100, show_plot=True, show_performances=True):
        """ 
        Tests the model and assesses the results

        Params:
        - num_data_to_plot: the number of value the be taken in account in the plot
        - show_plot: whether to plot the result
        - show_performances: whether to show the performances. The performances includes the RMSE and a detail of the precision of the forecasts
        """

        def plot_prediction_real_curves(y_pred):
            # Plot the results
            plot_size = num_data_to_plot
            x_plot = range(plot_size)

            plt.figure(figsize=(14,4))
            plt.plot(x_plot, self.y_test[:plot_size], color="#00FF00", label='Réalité')
            plt.plot(x_plot, y_pred[:plot_size], color="#002060", label='Prediction')
            plt.ylabel("Prix (€/MWh)")
            plt.xticks([])            
            plt.legend()
            plt.show()
        
        def display_performances(y_pred):

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

            y_pred_array = np.array(y_pred)
            y_test_array = np.array(self.y_test)

            error_array = (y_pred_array - y_test_array) / y_test_array * 100
            percent_value = 100 / len(error_array)            

            for error in error_array:

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
            
            error_df, positive_error_df = pd.DataFrame(error_dic, index=[0]), pd.DataFrame(positive_error_dic, index=[0])

            print("Number of tests :", len(self.y_test))

            rmse = root_mean_squared_error(self.y_test, y_pred)
            print(f"RMSE total : {rmse}")

            rmse_min = 10000000
            rmse_max = -1

            for day in range(len(self.y_test) // 24):
                day_rmse = root_mean_squared_error(self.y_test[ day * 24 : ( day + 1 ) * 24], y_pred[ day * 24 : ( day + 1 ) * 24])
                if day_rmse < rmse_min:
                    rmse_min = day_rmse
                if day_rmse > rmse_max:
                    rmse_max = day_rmse
            
            print(f"RMSE min : {rmse_min}")
            print(f"RMSE max : {rmse_max}", end='\n\n')

            print(error_df)
            print(positive_error_df)
        
        # Predictions on test set
        y_pred = self.model.predict(self.X_test)        
        
        if show_plot:
            plot_prediction_real_curves( y_pred )

        if show_performances:
            display_performances( y_pred )

    def get_rmse(self):
        y_pred = self.model.predict(self.X_test)
        return root_mean_squared_error(self.y_test, y_pred)
        