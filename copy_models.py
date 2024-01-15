import pandas as pd
import os
import shutil

if not os.path.exists('models_paramaters'):
    os.mkdir('models_paramaters')
rad_lines = [20,40,60,80,100]


for rl in rad_lines:
    df=pd.read_csv(f'results/results_{rl}lines.csv')
    print(df)
    for i, row in df.iterrows():
        path = row['model_path']
        filename = os.path.basename(path)
        shutil.copy(path, f'models_paramaters/{filename}')
        