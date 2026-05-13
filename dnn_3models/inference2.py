# get the .keras model
from tensorflow import keras
model = keras.models.load_model('models/dnn_model.keras')
#take the scaler
import joblib

imputer = joblib.load("models/imputer.joblib")
scaler = joblib.load("models/scaler.joblib")

#use the model to make predictions on the test set
import pandas as pd
from preprocess import preprocess_and_add_data, preprocess_data
from sklearn.preprocessing import StandardScaler

df = preprocess_and_add_data('test_x.csv')

print(df.size)

X_test_imputed = imputer.transform(df)
X_test_scaled = scaler.transform(X_test_imputed)

y_pred = model.predict(X_test_scaled)

print(y_pred.shape)
print(y_pred[:10])

#save the predictions to a csv file
submission = pd.DataFrame({'id': range(1, len(y_pred) + 1), 'bilissel_performans_skoru': y_pred.flatten()})
submission.to_csv('submission.csv', index=False)