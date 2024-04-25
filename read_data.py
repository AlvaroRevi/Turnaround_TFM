import pandas as pd
import numpy as np

filepath = '/Users/alvaroreviriego/Desktop/Turnaround_TFM/Turnaround_simplificado.xlsx'
sheet_name = 'LEBL_turnaround'
csv_path = '/Users/alvaroreviriego/Desktop/Turnaround_TFM/LEBL_turnaround.csv'
#raw_data = pd.read_excel(filepath, sheet_name=sheet_name)
raw_data = pd.read_csv(csv_path, delimiter=';')

print("End")
