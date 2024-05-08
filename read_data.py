import pandas as pd
import numpy as np
from openpyxl import load_workbook
from datetime import datetime

filepath = 'Turnaround_simplificado.xlsx'
sheet_name = 'LEBL_turnaround'
path_LEBL_turn = 'LEBL_turnaround.csv'
path_LEBL_dept = 'LEBL_departures.csv'
path_LEBL_arr = 'LEBL_arrivals.csv'
#raw_data = load_workbook(filepath, read_only=True)
#sheet = raw_data[sheet_name]


data_LEBL_turn = pd.read_csv(path_LEBL_turn, delimiter=";")
data_LEBL_dept = pd.read_csv(path_LEBL_dept, delimiter=";")
data_LEBL_arr = pd.read_csv(path_LEBL_arr, delimiter=";")
print("END")