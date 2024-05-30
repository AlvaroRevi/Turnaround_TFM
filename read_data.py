import pandas as pd
from functions_collection import *
import numpy as np
from datetime import datetime

'''
Este archivo es el archivo inicial que usamos para leer los ficheros csv que salen del 'Turnaround_simplificado.xlsx'
Una vez leido el archivo, se procesan los datos y se exportan a un csv las columnas que nos interesan. Estas columnas 
se pueden ir modificando a medida que avance el proyecto
'''


# Leemos el archivo con los datos de los aeropuertos
airports_data = pd.read_csv('airports_data.csv', delimiter=",")
# Nos quedamos unicamente con las columnas del dataframe que nos interesen
airports_data = airports_data[['ident','latitude_deg','longitude_deg']]
# Añadimos manualmente los aeropuertos que no salen en la lista
SPJC_airport = {'ident': 'SPJC','latitude_deg': -12.0219444, 'longitude_deg': -77.1144444}
LESJ_airport = {'ident': 'LESJ','latitude_deg': 39.5517 , 'longitude_deg':  2.7388}
LERJ_airport = {'ident': 'LERJ','latitude_deg': 42.4605556 , 'longitude_deg':  -2.3205555555555555}
new_entry = pd.DataFrame([SPJC_airport,LESJ_airport,LERJ_airport])

airports_data = pd.concat([airports_data,new_entry])

# Identificamos los archivos que queremos leer (meter en un bucle)
airports_list = ['LEBL','LEMD','LEMH','LEST']

# Identificamos el nombre del aeropuerto que estamos estudiando
for airport_turnaround in airports_list:

    path_turnaround_csv = airport_turnaround + '_turnaround.csv'
    output_name = airport_turnaround + '_turnaround_processed.csv'

    #----------------------------------------------------------------- LEBL ------------------------------------------------
    # Leemos el data frame entero
    data_turnaround_LEBL = pd.read_csv(path_turnaround_csv, delimiter=";")

    # Añadimos las coordenadas de arrival y destination y calculamos ambas distancias
    data_turnaround_LEBL = add_coordinates_and_distance(data_turnaround_LEBL, airports_data)

    # Añadimos los datetimes para poder hacer calculos
    data_turnaround_LEBL = compute_datetimes(data_turnaround_LEBL)

    # Calculamos los tiempos de 'Taxi In' y los tiempos de 'Taxi Out'
    data_turnaround_LEBL = compute_taxi_times(data_turnaround_LEBL)

    # Elegimos las columnas que queremos exportar
    columns_to_export = ['aerodrome','aircraftRegistration','aircraftType','arrivalAdep','departureAdes','airline',
                         'aldtDateTime','aibtDateTime','sobtDateTime','aobtDateTime','atotDateTime','TaxiInSeconds',
                         'TaxiOutSeconds','realTurnaroundSeconds','scheduleTurnaroundSeconds','arrivalLatitude',
                         'arrivalLongitude','departureLatitude','departureLongitude','arrivalDistance','departureDistance']

    # Exportamos las columnas del dataframe que nos interesen
    data_frame_to_export = data_turnaround_LEBL[columns_to_export]
    data_frame_to_export.to_csv(output_name,index = False)
print("END")