import pandas as pd
from functions_collection import *
import numpy as np
from datetime import datetime

# Importamos los datos de los csv procesados previamente
LEBL_df = pd.read_csv('LEBL_turnaround_processed.csv')
LEMD_df = pd.read_csv('LEMD_turnaround_processed.csv')
LEMH_df = pd.read_csv('LEMH_turnaround_processed.csv')
LEST_df = pd.read_csv('LEST_turnaround_processed.csv')
