import pandas as pd
from functions_collection import *
import numpy as np
from datetime import datetime

#data_LEBL_dept = pd.read_csv('LEBL_departures.csv', delimiter=";")
#data_LEBL_arr = pd.read_csv('LEBL_arrivals.csv', delimiter=";")


# Leemos el archivo con los datos de los aeropuertos
airports_data = pd.read_csv('airports_data.csv', delimiter=",")
# Nos quedamos unicamente con las columnas del dataframe que nos interesen
airports_data = airports_data[['ident','latitude_deg','longitude_deg']]

# Identificamos los archivos que queremos leer (meter en un bucle)
airports_name = ['LEBL','LEMD','LEMH','LEST']

# Identificamos el nombre del aeropuerto que estamos estudiando
airport_turnaround = airports_name[0] #Esta variable se cambia para hacer el bucle
path_turnaround_csv = airport_turnaround + '_turnaround.csv'
output_name = airport_turnaround + '_processed.csv'

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
                     'arrivalLongitude','departureLatitude','departureLongitude']

# Exportamos las columnas del dataframe que nos interesen
data_frame_to_export = data_turnaround_LEBL[columns_to_export]
#data_frame_to_export.to_csv(output_name,index = False)
print("END")