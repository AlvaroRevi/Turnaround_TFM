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

    data_turnaround['arrivalDistance'] = data_turnaround.apply(
        lambda row: geodesic((row['aerodromeLatitude'], row['aerodromeLongitude']), (row['arrivalLatitude'], row['arrivalLongitude'])).kilometers, axis=1)

    data_turnaround['departureDistance'] = data_turnaround.apply(
        lambda row: geodesic((row['aerodromeLatitude'], row['aerodromeLongitude']),
                             (row['departureLatitude'], row['departureLongitude'])).kilometers, axis=1)
    return data_turnaround

def compute_datetimes(data_turnaround):
    '''
    Añade una columna de fecha y hora combinadas para poder hacer calculos
    :param data_turnaround:
    :return: data_turnaround
    '''
    data_turnaround['aldtDateTime'] = pd.to_datetime(data_turnaround['aldtDate']+' '+data_turnaround['aldtTime'],
                                                     format='%d/%m/%Y %H:%M:%S')
    data_turnaround['aibtDateTime'] = pd.to_datetime(data_turnaround['aibtDate'] + ' ' + data_turnaround['aibtTime'],
                                                     format='%d/%m/%Y %H:%M:%S')
    data_turnaround['sibtDateTime'] = pd.to_datetime(data_turnaround['sibtDate'] + ' ' + data_turnaround['sibtTime'],
                                                     format='%d/%m/%Y %H:%M:%S')
    data_turnaround['sobtDateTime'] = pd.to_datetime(data_turnaround['sobtDate'] + ' ' + data_turnaround['sobtTime'],
                                                     format='%d/%m/%Y %H:%M:%S')
    data_turnaround['aobtDateTime'] = pd.to_datetime(data_turnaround['aobtDate'] + ' ' + data_turnaround['aobtTime'],
                                                     format='%d/%m/%Y %H:%M:%S')
    data_turnaround['atotDateTime'] = pd.to_datetime(data_turnaround['atotDate'] + ' ' + data_turnaround['atotTime'],
                                                     format='%d/%m/%Y %H:%M:%S')

    return data_turnaround

def compute_taxi_times(data_turnaround):
    '''
    Calcula los tiempos de taxi al entrar y salir de pista
    :param data_turnaround:
    :return:
    '''

    data_turnaround['TaxiInSeconds'] = ((data_turnaround['aibtDateTime'] - data_turnaround['aldtDateTime']).dt.total_seconds())
    data_turnaround['TaxiOutSeconds'] = ((data_turnaround['atotDateTime'] - data_turnaround['aobtDateTime']).dt.total_seconds())
    return data_turnaround

