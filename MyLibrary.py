import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn import preprocessing

from sklearn import linear_model

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score as EV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE


from sklearn.model_selection import train_test_split as TTS





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


def fit_linear_reg(X,Y):    
    model_k = linear_model.LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = MSE(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared

def feature_selection(X_fs, y_fs):
    k = len(X_fs.columns)
    remaining_features = list(X_fs.columns.values)
    features = []
    RSS_list, R_squared_list = [np.inf], [np.inf] 
    features_list = dict()

    for i in range(1,k+1):
        best_RSS = np.inf
        
        for combo in itertools.combinations(remaining_features, 1):

                RSS = fit_linear_reg(X_fs[list(combo) + features], y_fs)   #Store temp result 

                if RSS[0] < best_RSS:
                    best_RSS = RSS[0]
                    best_R_squared = RSS[1] 
                    best_feature = combo[0]

        #Updating variables for next loop
        features.append(best_feature)
        remaining_features.remove(best_feature)
        
        #Saving values for plotting
        RSS_list.append(best_RSS)
        R_squared_list.append(best_R_squared)
        features_list[i] = features.copy()

    # Save the RSS and R squared.
    df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
    df1['numb_features'] = df1.index
    return df1

def plot_feature_selection_criterion(df, X_fs, y_fs):
    # plot model selection criteria against the model complexity.
    p = len(X_fs.columns)
    m = len(y_fs)
    hat_sigma_squared = (1/(m - p -1)) * min(df['RSS'])
    df['C_p'] = (1/m) * (df['RSS'] + 2 * df['numb_features'] * hat_sigma_squared )
    df['AIC'] = (1/(m*hat_sigma_squared)) * (df['RSS'] + 2 * df['numb_features'] * hat_sigma_squared )
    df['BIC'] = (1/(m*hat_sigma_squared)) * (df['RSS'] +  np.log(m) * df['numb_features'] * hat_sigma_squared )
    df['R_squared_adj'] = 1 - ( (1 - df['R_squared'])*(m-1)/(m-df['numb_features'] -1))


    variables = ['C_p', 'AIC','BIC','R_squared_adj']
    fig = plt.figure(figsize = (18,6))

    for i,v in enumerate(variables):
        ax = fig.add_subplot(1, 4, i+1)
        ax.plot(df['numb_features'],df[v], color = 'lightblue')
        ax.scatter(df['numb_features'],df[v], color = 'darkblue')
        if v == 'R_squared_adj':
            ax.plot(df[v].idxmax(),df[v].max(), marker = 'x', markersize = 20)
        else:
            ax.plot(df[v].idxmin(),df[v].min(), marker = 'x', markersize = 20)
        ax.set_xlabel('Number of predictors')
        ax.set_ylabel(v)

    fig.suptitle('Subset selection using C_p, AIC, BIC, Adjusted R2', fontsize = 16)
    plt.show()
# print("---------------------------------------")
# print("---------------------------------------")
# print(f"\nModel: {r.activation} \nLayers: {r.n_layers_} \nSize: {s} \nMSE: {MSE(y_test, y_pred)}\nR^2 Score: {round(r2_score(y_test, y_pred), 2)}\nExplained Variace: {round(EV(y_test, y_pred),2)}\nMean Absolute Error: {round(MAE(y_test, y_pred),2)}")
#         pd.DataFrame(r.loss_curve_).plot(title = f"Model: {r.activation} \nLayers: {r.n_layers_}")
#         print("\n")