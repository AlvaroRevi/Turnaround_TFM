import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
'''
Este script es una replica del modelo numero 2 en el que se filtran unicamente los datos de vuelos para la aerolinea
Iberia (IBE), lo cual puede ayudarnos de cara a una implementación estrategica
'''
# Importamos los datos de los csv procesados previamente

# LEBL_df = pd.read_csv('LEBL_turnaround_processed.csv')
LEMD_df = pd.read_csv('..\Data\LEMD_turnaround_processed.csv')
# LEMH_df = pd.read_csv('LEMH_turnaround_processed.csv')
# LEST_df = pd.read_csv('LEST_turnaround_processed.csv')

data = LEMD_df

# Convertir fecha y hora
date_columns = ['aldtDateTime', 'aibtDateTime', 'sobtDateTime', 'aobtDateTime', 'atotDateTime']
for col in date_columns:
    data[col] = pd.to_datetime(data[col])

# Crear características adicionales a partir de las fechas
data['aldt_month'] = data['aldtDateTime'].dt.month
data['aldt_day_of_week'] = data['aldtDateTime'].dt.dayofweek
data['aldt_hour'] = data['aldtDateTime'].dt.hour

data['aibt_month'] = data['aibtDateTime'].dt.month
data['aibt_day_of_week'] = data['aibtDateTime'].dt.dayofweek
data['aibt_hour'] = data['aibtDateTime'].dt.hour

data['sobt_month'] = data['sobtDateTime'].dt.month
data['sobt_day_of_week'] = data['sobtDateTime'].dt.dayofweek
data['sobt_hour'] = data['sobtDateTime'].dt.hour

data['aobt_month'] = data['aobtDateTime'].dt.month
data['aobt_day_of_week'] = data['aobtDateTime'].dt.dayofweek
data['aobt_hour'] = data['aobtDateTime'].dt.hour

data['atot_month'] = data['atotDateTime'].dt.month
data['atot_day_of_week'] = data['atotDateTime'].dt.dayofweek
data['atot_hour'] = data['atotDateTime'].dt.hour


# Codificacion de variables categoricas
data = data[data['airline'] == 'EZY']
data = pd.get_dummies(data, columns = ['aircraftRegistration','aircraftType'])



numerical_features = ['TaxiInSeconds','TaxiOutSeconds','realTurnaroundSeconds','scheduleTurnaroundSeconds',
                      'arrivalLatitude','arrivalLongitude','departureLatitude','departureLongitude','arrivalDistance',
                      'departureDistance','aldt_month', 'aldt_day_of_week', 'aldt_hour','aibt_month','aibt_day_of_week',
                      'aibt_hour','sobt_month', 'sobt_day_of_week', 'sobt_hour','aobt_month', 'aobt_day_of_week',
                      'aobt_hour','atot_month', 'atot_day_of_week', 'atot_hour']
scaler = StandardScaler()

data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Seleccion de caracteristicas

X = data.drop(columns=['aerodrome','airline','arrivalAdep','departureAdes','realTurnaroundSeconds',
                       'aldtDateTime','aibtDateTime','sobtDateTime','aobtDateTime','atotDateTime','TaxiInSeconds',
                       'TaxiOutSeconds','arrivalLatitude','arrivalLongitude','departureLatitude','departureLongitude'])

y = data['realTurnaroundSeconds']

# Division de datos

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#------------- KERAS MODEL ---------------
model_path = 'model_tensorflow_filter_EZY.keras'

model = tf.keras.models.load_model(model_path)

y_pred_tf = model.predict(X_test)
mse_tf = mean_squared_error(y_test, y_pred_tf)
rmse_tf = root_mean_squared_error(y_test,y_pred_tf)
mae_tf = mean_absolute_error(y_test, y_pred_tf)
r2_tf = r2_score(y_test, y_pred_tf)

print(f'Loaded MSE: {mse_tf}')
print(f'Loaded RMSE:  {rmse_tf}')
print(f'Loaded MAE: {mae_tf}')
print(f'Loaded R²: {r2_tf}')