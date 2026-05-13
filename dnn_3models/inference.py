#use joblib to load models and make predictions
import joblib
import csv
import pandas as pd
from preprocess import preprocess_data

df = preprocess_data('test_x.csv')
x_test = df.drop('id', axis=1) #drop id column

#load each model and make predictions
for model_name in ['RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor']:
    model = joblib.load(f'models/{model_name}.joblib')
    #make predictions
    y_pred = model.predict(x_test)
    #save predictions to csv file
    with open(f'predictions_{model_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'bilissel_performans_skoru'])
        for i, pred in enumerate(y_pred):
            writer.writerow([i, pred])
    