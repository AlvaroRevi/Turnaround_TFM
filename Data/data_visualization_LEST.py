import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

'''
En este script utilizamos un modelo de redes neuronales Sequential con los siguientes features:
        -TaxiInSeconds
        -TaxiOutSeconds
        -scheduleTurnaroundSeconds
        -arrivalDistance
        -departureDistance
        -Actual dates and times
        -aircraftRegistration
        -airline 
        -aircraftType
'''
# Importamos los datos de los csv procesados previamente

# LEBL_df = pd.read_csv('..\Data\LEBL_turnaround_processed.csv')
#LEMD_df = pd.read_csv('..\Data\LEMD_turnaround_processed.csv')
# LEMH_df = pd.read_csv('..\Data\LEMH_turnaround_processed.csv')
LEST_df = pd.read_csv('..\Data\LEST_turnaround_processed.csv')

data = LEST_df

# Convertir fecha y hora
date_columns = ['aldtDateTime', 'aibtDateTime', 'sobtDateTime', 'aobtDateTime', 'atotDateTime']
for col in date_columns:
    data[col] = pd.to_datetime(data[col])


# Codificacion de variables categoricas
data = pd.get_dummies(data, columns = ['aircraftRegistration','aircraftType','airline'])

num_entradas = data['realTurnaroundSeconds'].count()
mean_turnaround = data['realTurnaroundSeconds'].mean()
std_turnaround = data['realTurnaroundSeconds'].std()
quantiles_turnaround = data['realTurnaroundSeconds'].quantile([0.25,0.5,0.75])

print(f'Numero de entradas: {num_entradas}')
print(f'Media: {mean_turnaround}')
print(f'Std: {std_turnaround}')
print(f'Cuartil 25: {quantiles_turnaround[0.25]}')
print(f'Cuartil 50: {quantiles_turnaround[0.5]}')
print(f'Cuartil 75: {quantiles_turnaround[0.75]}')


#
plt.figure(figsize=(10, 6))
sns.histplot(data['realTurnaroundSeconds'], bins=100, kde=True, color='grey')
plt.title('Output variable distribution in LEMH airport')
plt.xlabel('Turnaround Time (seconds)')
plt.ylabel('Frecuency')
plt.show()

