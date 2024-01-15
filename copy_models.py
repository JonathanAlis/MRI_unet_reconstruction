import pandas as pd
import os
import shutil

os.mkdir('models_paramaters')
rad_lines = [20,40,60,80,100]


for rl in rad_lines:
    df=pd.read_excel(f'results_{rl}lines.xlsx')
    for i, row in df.iterrows():
        path = row['model_path']
        filename = os.path.basename(path)
        shutil.copy(row, f'models_paramaters/{filename}')
        