import pandas as pd
from functions_collection import *
import numpy as np
from datetime import datetime

#data_LEBL_dept = pd.read_csv('LEBL_departures.csv', delimiter=";")
#data_LEBL_arr = pd.read_csv('LEBL_arrivals.csv', delimiter=";")


# Leemos
airports_data = pd.read_csv('airports_data.csv', delimiter=",")
# Nos quedamos unicamente con las columnas del dataframe que nos interesen
airports_data = airports_data[['ident','latitude_deg','longitude_deg']]
#----------------------------------------------------------------- LEBL ------------------------------------------------
# Leemos el data frame entero
data_turnaround_LEBL = pd.read_csv('LEBL_turnaround.csv', delimiter=";")

# Añadimos las coordenadas de arrival y destination y calculamos ambas distancias
data_turnaround_LEBL = add_coordinates_and_distance(data_turnaround_LEBL, airports_data)




print("END")