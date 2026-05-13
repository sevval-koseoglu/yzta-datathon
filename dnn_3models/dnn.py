from preprocess import preprocess_and_add_data, preprocess_data
from model import create_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
import numpy as np
df = preprocess_and_add_data('train.csv')

#split the data into features and target variable, last column is target variable
#there are headers in the csv file, so we can use them to select the features and target variable
x = df.drop('bilissel_performans_skoru', axis=1)
y = df['bilissel_performans_skoru']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


dnn_model, reduce_lr = create_model((x_train.shape[1],))
print('Training DNN...')


# Imputer sadece train üzerinde fit edilmeli
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(x_train)
X_test_imputed = imputer.transform(x_test)
import joblib
joblib.dump(imputer, "models/imputer.joblib")

# Scaler da sadece train üzerinde fit edilmeli
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

#save the scaler


joblib.dump(scaler, 'models/scaler.joblib')


history = dnn_model.fit(X_train_scaled, y_train, 
                        epochs=200,
                        batch_size=128, 
                        validation_data=(X_test_scaled, y_test), 
                        callbacks=[reduce_lr])

import matplotlib.pyplot as plt

# Graph 1: Loss
plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["rmse"], label="train RMSE")
plt.plot(history.history["val_rmse"], label="validation RMSE")
plt.title("RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.show()

y_pred = np.clip(dnn_model.predict(X_test_scaled), 0, 10)
rmse = root_mean_squared_error(y_test, y_pred)
print(f'DNN RMSE: {rmse}')

#save dnn model
dnn_model.save('models/dnn_model.keras')

#save the scaler
import joblib
joblib.dump(scaler, 'models/scaler.joblib')