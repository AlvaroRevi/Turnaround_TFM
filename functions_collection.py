import numpy as np
import pandas as pd
from geopy.distance import geodesic
# def calculate_arrival_distance(aerodrome_lat, aerodrome_lon, arrival_lat, arrival_lon):
#     # aerodrome_coords = (data_turnaround['aerodromeLatitude'],data_turnaround['aerodromeLongitude'])
#     # arrival_coords = (data_turnaround['arrivalLatitude'], data_turnaround['arrivalLongitude'])
#     aerodrome_coords = (aerodrome_lat, aerodrome_lon)
#     arrival_coords = (arrival_lat,arrival_lon)
#     return geodesic(aerodrome_coords, arrival_coords).kilometers
def add_coordinates_and_distance(data_turnaround, airports_data):
    # Añadimos las coordenadas de los aeropuertos usando el dataframe "airports data", para eso, seguimos los siguientes pasos
    # Añadimos al dataframe inicial los datos de "airports_data" basados en la coincidencia del codigo ICAO de la columna
    # 'arrivalAdep' del data_turnaround con la columna del airports_data de 'ident'

    data_turnaround = pd.merge(data_turnaround, airports_data, left_on='arrivalAdep', right_on='ident',
                                    how='left')

    # Eliminamos la columna 'ident' (la que contiene el codigo ICAO) porque al ser la columna coincidente la tenemos repetida
    data_turnaround.drop(columns=['ident'], inplace=True)

    # Cambiamos el nombre de la latitud y la longitud a la que estemos importando, en este caso la de arrival
    data_turnaround.rename(columns={'latitude_deg': 'arrivalLatitude', 'longitude_deg': 'arrivalLongitude'},
                                inplace=True)

    # Repetimos el proceso para las coordenadas de los departures y del propio aerodromo
    data_turnaround = pd.merge(data_turnaround, airports_data, left_on='departureAdes', right_on='ident',
                                    how='left')
    data_turnaround.drop(columns=['ident'], inplace=True)
    data_turnaround.rename(columns={'latitude_deg': 'departureLatitude', 'longitude_deg': 'departureLongitude'},
                                inplace=True)
    data_turnaround = pd.merge(data_turnaround, airports_data, left_on='aerodrome', right_on='ident',
                                    how='left')
    data_turnaround.drop(columns=['ident'], inplace=True)
    data_turnaround.rename(columns={'latitude_deg': 'aerodromeLatitude', 'longitude_deg': 'aerodromeLongitude'},
                                inplace=True)

    # Llena los NaN de ceros
    data_turnaround['aerodromeLatitude'] = data_turnaround['aerodromeLatitude'].fillna(0)
    data_turnaround['aerodromeLongitude'] = data_turnaround['aerodromeLongitude'].fillna(0)
    data_turnaround['arrivalLatitude'] = data_turnaround['arrivalLatitude'].fillna(0)
    data_turnaround['arrivalLongitude'] = data_turnaround['arrivalLongitude'].fillna(0)
    data_turnaround['departureLatitude'] = data_turnaround['departureLatitude'].fillna(0)
    data_turnaround['departureLongitude'] = data_turnaround['departureLongitude'].fillna(0)

    aerodrome_coords = (data_turnaround['aerodromeLatitude'],data_turnaround['aerodromeLongitude'])
    arrival_coords = (data_turnaround['arrivalLatitude'], data_turnaround['arrivalLongitude'])
    departure_coords = (data_turnaround['departureLatitude'], data_turnaround['departureLongitude'])

    data_turnaround['arrivalDistance'] = data_turnaround.apply(
        lambda row: geodesic((row['aerodromeLatitude'], row['aerodromeLongitude']), (row['arrivalLatitude'], row['arrivalLongitude'])).kilometers, axis=1)

    data_turnaround['departureDistance'] = data_turnaround.apply(
        lambda row: geodesic((row['aerodromeLatitude'], row['aerodromeLongitude']),
                             (row['departureLatitude'], row['departureLongitude'])).kilometers, axis=1)
    return data_turnaround
