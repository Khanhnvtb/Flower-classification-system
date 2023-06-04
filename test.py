
import json
import numpy as np
from PIL import Image
from Feature import Feature
import os

# Đọc dữ liệu từ file JSON
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

feature = Feature()
# Lấy dữ liệu'
feature_data = data['data']


folder_path = './Flower/'
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

count = 0
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
            distance = distance_color +  distance_hog
            
            list_distance.append((label, distance))

            color_distance.append(distance_color)
            hog_distance.append(distance_hog)

        list_distance.sort(key=lambda x: x[1])
        list_distance = list_distance[:5]
        list_label = [x[0] for x in list_distance]
        # get element frequency max in list
        label_count = {}
        for label, distance in list_distance:
            if label in label_count:
                label_count[label]['count'] += 1
                if distance < label_count[label]['distance']:
                    label_count[label]['distance'] += distance
            else:
                label_count[label] = {
                    'count': 1,
                    'distance': distance
                }

        label_count = sorted(label_count.items(), key=lambda x: (-x[1]['count'],x[1]['distance']))
        most_common_label = label_count[0][0]

        if most_common_label == subfolder:
            count += 1


print(count)