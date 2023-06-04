import json 
import numpy as np
import pandas as pd
# Đọc dữ liệu từ file JSON

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    data = data['data']

list_color_distance = []
list_hog_distance = []

for i in range(len(data)):
    for j in range(i+1, len(data)):
        color_feature1 = np.array(data[i]['color_feature'])
        color_feature2 = np.array(data[j]['color_feature'])
        hog_feature1 = np.array(data[i]['hog_feature'])
        hog_feature2 = np.array(data[j]['hog_feature'])

        distance_color = np.linalg.norm(color_feature1 - color_feature2)
        distance_hog = np.linalg.norm(hog_feature1 - hog_feature2)

        list_color_distance.append(distance_color)
        list_hog_distance.append(distance_hog)

df = pd.DataFrame({'color_distance': list_color_distance, 'hog_distance': list_hog_distance})
# print(df.describe().loc[['mean', 'std', 'min', 'max']])
print(df.describe())