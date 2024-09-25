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
LEMD_df = pd.read_csv('..\Data\LEMD_turnaround_processed.csv')
# LEMH_df = pd.read_csv('..\Data\LEMH_turnaround_processed.csv')
# LEST_df = pd.read_csv('..\Data\LEST_turnaround_processed.csv')

data = LEMD_df

# Convertir fecha y hora
date_columns = ['aldtDateTime', 'aibtDateTime', 'sobtDateTime', 'aobtDateTime', 'atotDateTime']
for col in date_columns:
    data[col] = pd.to_datetime(data[col])


# Codificacion de variables categoricas
data = pd.get_dummies(data, columns = ['aircraftRegistration','aircraftType','airline'])

17


