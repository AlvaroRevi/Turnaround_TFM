import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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
data = data[data['airline'] == 'IBE']
data = pd.get_dummies(data, columns = ['aircraftRegistration','aircraftType'])



numerical_features = ['TaxiInSeconds','TaxiOutSeconds','scheduleTurnaroundSeconds',
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
scaler_y = StandardScaler()
y_reshaped = y.values.reshape(-1,1)
y_scaled = scaler_y.fit_transform(y_reshaped)

# Division de datos

X_train, X_test, y_train, y_test = train_test_split(X,y_scaled,test_size=0.2, random_state=42)

#------------- KERAS MODEL ---------------
model_path = 'model_tensorflow_filter_IBE.keras'

model = tf.keras.models.load_model(model_path)

y_pred = model.predict(X_test)

y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred)

mse = mean_squared_error(y_test_original, y_pred_original)
rmse = root_mean_squared_error(y_test_original,y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f'Loaded MSE: {mse}')
print(f'Loaded RMSE: {rmse}')
print(f'Loaded MAE: {mae}')
print(f'Loaded R²: {r2}')

# Histograma de los valores reales
plt.subplot(1, 2, 1)
sns.histplot(y_test_original, kde=True, color='blue', bins=30)
plt.title('Actual value distribution')
plt.xlabel('Turnaround Time (seconds)')
plt.ylabel('Frecuency')
plt.ylim(0,3000)

# Histograma de las predicciones
plt.subplot(1, 2, 2)
sns.histplot(y_pred_original.flatten(), kde=True, color='orange', bins=30)
plt.title('Predicted value distribution')
plt.xlabel('Turnaround Time (seconds)')
plt.ylabel('Frecuency')
plt.ylim(0,3000)
plt.show()


# Seleccionar los primeros 1000 puntos


y_test_subset = y_test_original[:100]
y_pred_subset = y_pred_original[:100]

# Crear el plot
plt.figure(figsize=(14, 7))
plt.plot(y_test_subset, label='Actual values', color='blue')
plt.plot(y_pred_subset, label='Predicted values', color='orange', linestyle='dashed')
plt.title('Sample comparative between actual and predicted values')
plt.xlabel('Index')
plt.ylabel('Turnaround Time (segundos)')
plt.legend()
plt.show()

# Distribución de los errores (residuals)
residuals = y_test_original.flatten() - y_pred_original.flatten()

plt.figure(figsize=(7, 6))
sns.histplot(residuals, kde=True, color='green', bins=200)
plt.title('Residual errors distribution')
plt.xlabel('Prediction error (seconds)')
plt.ylabel('Frecuency')
plt.show()