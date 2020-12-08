import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score as EV
from sklearn.metrics import mean_absolute_error as MAE

from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE




def predRespPlots(X, y, save = False, path = "", file_type = ".png"):
    col = X.columns.drop(y)
    for c in col:
        if isinstance(X[c][0], str):
            X.boxplot(y, by = c)
            if save:
                file = f"{path}/{y}_by_{c}" + file_type
                plt.savefig(f"{file}")
        else:
            fig = plt.figure()
            plt.scatter(X[c], X[y])
            plt.xlabel(c)
            plt.ylabel(y)
            
            if save:
                file = f"{path}/{y}_by_{c}" + file_type
                fig.savefig(f"{file}")
        plt.show()




# print("---------------------------------------")
# print("---------------------------------------")
# print(f"\nModel: {r.activation} \nLayers: {r.n_layers_} \nSize: {s} \nMSE: {MSE(y_test, y_pred)}\nR^2 Score: {round(r2_score(y_test, y_pred), 2)}\nExplained Variace: {round(EV(y_test, y_pred),2)}\nMean Absolute Error: {round(MAE(y_test, y_pred),2)}")
#         pd.DataFrame(r.loss_curve_).plot(title = f"Model: {r.activation} \nLayers: {r.n_layers_}")
#         print("\n")