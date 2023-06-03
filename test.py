
import json
import numpy as np
from PIL import Image
from Feature import Feature
import os

# Đọc dữ liệu từ file JSON
with open('data.json', 'r') as f:
    data = json.load(f)

feature = Feature()
# Lấy dữ liệu'
feature_data = data['data']


folder_path = './Flower/'
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]


list_five = []
for subfolder in subfolders:
    files = os.listdir(os.path.join(folder_path, subfolder))
    
    for file_name in files:
        img = Image.open(os.path.join(folder_path, subfolder, file_name))
        img = np.array(img)

        color_feature = feature.color_histogram(img)
        hog_feature = feature.hog(img)

        list_distance = []
        color_distance = []
        hog_distance = []

        for data in feature_data:
            label = data['label']
            color_feature2 = np.array(data['color_feature'])
            hog_feature2 = np.array(data['hog_feature'])
            
            distance_color = feature.distanceEuclidean(color_feature, color_feature2)
            distance_hog = feature.distanceEuclidean(hog_feature, hog_feature2)
            distance = 0.8 * distance_color + 0.2 * distance_hog
            
            list_distance.append((label, distance))

            color_distance.append(distance_color)
            hog_distance.append(distance_hog)

        list_distance.sort(key=lambda x: x[1])
        

        with open('result.txt', 'a', encoding='utf-8') as f:
            f.write(f'{list_distance[1][1]} - {list_distance[6][1]}\n')

print(list_five)