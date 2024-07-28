import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# Importamos los datos de los csv procesados previamente

# LEBL_df = pd.read_csv('LEBL_turnaround_processed.csv')
LEMD_df = pd.read_csv('.\Data\LEMD_turnaround_processed.csv')
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
data = pd.get_dummies(data, columns = ['aircraftRegistration','aircraftType','airline'])

# encoder = OneHotEncoder(sparse = False)
# encoded_categorical = encoder.fit_transform(data[['aircraftRegistration','aircraftType','airline']])
# encoded_df = pd.DataFrame(enconded_categorical, columns=encoder.get_feature_names_out(['aircraftRegistration','aircraftType','airline']))



numerical_features = ['TaxiInSeconds','TaxiOutSeconds','scheduleTurnaroundSeconds',
                      'arrivalLatitude','arrivalLongitude','departureLatitude','departureLongitude','arrivalDistance',
                      'departureDistance','aldt_month', 'aldt_day_of_week', 'aldt_hour','aibt_month','aibt_day_of_week',
                      'aibt_hour','sobt_month', 'sobt_day_of_week', 'sobt_hour','aobt_month', 'aobt_day_of_week',
                      'aobt_hour','atot_month', 'atot_day_of_week', 'atot_hour']



# Selección de características (excluyendo 'realTurnaroundSeconds' y las columnas de fecha/hora originales)
X = data.drop(columns=['aerodrome','arrivalAdep','departureAdes','realTurnaroundSeconds', 'aldtDateTime', 'aibtDateTime',
                       'sobtDateTime', 'aobtDateTime', 'atotDateTime'])


# Normalizar las características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_features])





# Aplicar PCA
pca = PCA(n_components=10)  # Puedes ajustar el número de componentes según sea necesario
pca.fit(X_scaled)

# Varianza explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_
print(f'Varianza explicada por cada componente principal: {explained_variance}')

# Cargar de las características en las componentes principales
components = pca.components_

# Crear un DataFrame con las cargas
component_df = pd.DataFrame(components, columns=numerical_features)

# Visualización de la varianza explicada acumulada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada por el PCA')
plt.grid(True)
plt.show()

# Visualizar las cargas de las características en las primeras dos componentes principales
plt.figure(figsize=(12, 6))
plt.bar(x=range(len(numerical_features)), height=component_df.iloc[0].abs(), tick_label=numerical_features)
plt.xticks(rotation=90)
plt.xlabel('Características')
plt.ylabel('Cargas de la Primer Componente Principal')
plt.title('Cargas de las Características en la Primer Componente Principal')
plt.show()
