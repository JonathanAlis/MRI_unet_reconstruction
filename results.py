import os
import re
import pandas as pd
from itertools import combinations, product
from scipy.stats import f_oneway
#from statsmodels.stats.multicomp import pairwise_tukeyhsd



unet_types = ['Unet', 'ResnetUnet', 'ConvUNeXt']
type_rec = ['L2', 'L1', 'TV']
type_recs = []
for r in range(1, len(type_rec) + 1):
    for combination in combinations(type_rec, r):
        type_recs.append('_'.join(combination))
print(type_recs)
print(unet_types)
radial_lines_num = [20,40,60,80,100]
print(radial_lines_num)


all_combinations = list(product(unet_types, radial_lines_num, type_recs))
df = pd.DataFrame(all_combinations, columns=['unet', 'radial_lines', 'reconstruction'])
print(df)

results_path = 'result_images/'
folders = os.listdir(results_path)
folders = [f for f in folders if f.endswith('epochs')]
lists_all = []#lists1, lists2, lists3, ...]

for i, row in df.iterrows():
    if 1:
        unet = row['unet']
        rl = row['radial_lines']
        rec = row['reconstruction']
        folder_start = f'{unet}_{rl}lines_{rec}'
        folder_name = [results_path+f for f in folders if f.startswith(folder_start)][0]
        match = re.search(r'\d+', folder_name[::-1])
        epochs = int(match.group()[::-1])
        df.at[i, 'epochs'] = int(epochs)
        df.at[i, 'folder'] = folder_name
        
        print(folder_name)
        csv = f'{folder_name}/metrics.csv'
        csv_df = pd.read_csv(csv)
        
        avg_psnr = csv_df['psnr'].mean()
        std_psnr = csv_df['psnr'].std()
        avg_ssim = csv_df['ssim'].mean()
        std_ssim = csv_df['ssim'].std()
        avg_lpips = csv_df['lpips'].mean()
        std_lpips = csv_df['lpips'].std()
        metrics = f'{avg_psnr:.2f} / {avg_ssim:.3f} / {avg_lpips:.3f}'
        metrics_std = f'{avg_psnr:.2f}({std_psnr:.2f}) / {avg_ssim:.3f}({std_ssim:.3f}) / {avg_lpips:.3f}({std_lpips:.3f})'

        df.at[i, 'psnr'] = avg_psnr
        df.at[i, 'ssim'] = avg_ssim
        df.at[i, 'lpips'] = avg_lpips
        df.at[i, 'metrics'] = metrics
        df.at[i, 'metrics_std'] = metrics_std

        model_path = f'MRIrec_experiments/{unet}_{rl}lines_{rec}/{unet}_{rl}lines_{rec}_epoch{epochs}_constantLR_0.001.pth'
        df.at[i, 'model_path'] = model_path
        lists_all.append(list(csv_df['psnr']))

# Example: Assuming lists1, lists2, lists3, ... are your lists of values

stat, p_value = f_oneway(*lists_all)
print(stat, p_value)

        #print(csv_df)
    #except:
    #    pass
decimal_places_dict = {'epocs': 0, 'psnr': 2, 'ssim': 3, 'lpips': 3}
df = df.round(decimal_places_dict)
#print(df)
df.to_excel('results.xlsx', index=False)
grouped = df.groupby('radial_lines')
result_dict = {key: group for key, group in grouped}
#print(result_dict)
for key, value in result_dict.items():
    #print(f"\nKey: {key}\n{value}")
    value.to_excel(f'results_{key}lines.xlsx', index=False)
    value.drop(columns=['radial_lines','epochs','folder'], inplace=True)

    df_rl = value.pivot(index=['reconstruction'], columns=['unet'], values = 'metrics')#['psnr','ssim','lpips'])

    df_rl = df_rl.loc[type_recs]
    df_rl = df_rl[unet_types]
    #print(df_rl)
    df_rl.to_excel(f'formated_results_{key}lines.xlsx', index=True)
    print(df_rl.to_latex())


#------
grouped = df.groupby('unet')
result_dict = {key: group for key, group in grouped}
#print(result_dict)

for key, value in result_dict.items():
    #print(f"\nKey: {key}\n{value}")
    value.to_excel(f'results_{key}.xlsx', index=False)
    value.drop(columns=['unet','epochs','folder'], inplace=True)

    df_rl = value.pivot(index=['reconstruction'], columns=['radial_lines'], values = 'metrics')#['psnr','ssim','lpips'])

    df_rl = df_rl.loc[type_recs]
    df_rl = df_rl[radial_lines_num]
    #print(df_rl)
    df_rl.to_excel(f'formated_results_{key}.xlsx', index=True)
    print(df_rl.to_latex())
          
print(csv_df.head(10))
results_subset = [f for f in csv_df.head(10)['id']]
counter = 0
with open('subset_path.txt', 'w') as file:
    for r in results_subset:
        for f in folders:
            match = re.search(r'\d+', f)
            if match and int(match.group()) < 30:                
                path = f'result_images/{f}/{r}.png'
                file.write(path + '\n')
                counter+=1
        for tr in type_rec:
            path = f'BIRN_dataset/birn_pngs_20lines_{tr}/{r}_{tr}_20lines.png'
            file.write(path + '\n')
            counter+=1
                
print(folders)
print(counter)
#result_images/ConvUNeXt_20lines_TV_100epochs

