import pandas as pd
from functions_collection import *
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importamos los datos de los csv procesados previamente
# LEBL_df = pd.read_csv('LEBL_turnaround_processed.csv')
LEMD_df = pd.read_csv('LEMD_turnaround_processed.csv')
# LEMH_df = pd.read_csv('LEMH_turnaround_processed.csv')
# LEST_df = pd.read_csv('LEST_turnaround_processed.csv')

data = LEMD_df

# Convertir fecha y hora
date_columns = ['aldtDateTime', 'aibtDateTime', 'sobtDateTime', 'aobtDateTime', 'atotDateTime']
for col in date_columns:
    data[col] = pd.to_datetime(data[col])


# Codificacion de variables categoricas
data = pd.get_dummies(data, columns = ['aircraftRegistration','aircraftType','airline'])



numerical_features = ['TaxiInSeconds','TaxiOutSeconds','realTurnaroundSeconds','scheduleTurnaroundSeconds',
                      'arrivalLatitude','arrivalLongitude','departureLatitude','departureLongitude','arrivalDistance',
                      'departureDistance']
scaler = StandardScaler()

data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Seleccion de caracteristicas

X = data.drop(columns=['aerodrome','arrivalAdep','departureAdes','realTurnaroundSeconds',
                       'aldtDateTime','aibtDateTime','sobtDateTime','aobtDateTime','atotDateTime'])
y = data['realTurnaroundSeconds']

# Division de datos

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')
print("END")
