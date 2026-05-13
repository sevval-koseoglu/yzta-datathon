from preprocess import preprocess_and_add_data, preprocess_data
from model import create_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = preprocess_and_add_data('train.csv')
#split the data into features and target variable, last column is target variable
#there are headers in the csv file, so we can use them to select the features and target variable
x = df.drop('bilissel_performans_skoru', axis=1)
y = df['bilissel_performans_skoru']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Train a machine learning model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import HuberRegressor, ElasticNet
model_list = [RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor(), HuberRegressor(max_iter=10000), ElasticNet()]
trained_models = []

for model in model_list:
    print(f'Training {model.__class__.__name__}...')
    model.fit(x_train, y_train)   
    y_pred = model.predict(x_test)
    from sklearn.metrics import root_mean_squared_error
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f'{model.__class__.__name__} RMSE: {rmse}') #rmse
    trained_models.append(model)


#compare the models and select the best one based on RMSE
best_model = min(trained_models, key=lambda x: root_mean_squared_error(y_test, x.predict(x_test)))
print(f'Best model: {best_model.__class__.__name__}')


#save all models under models/
import joblib
for model in trained_models:
    #save each by joblib except for dnn model, because it is a keras model
    if model.__class__.__name__ != 'Sequential':
        joblib.dump(model, f'models/{model.__class__.__name__}.joblib')











