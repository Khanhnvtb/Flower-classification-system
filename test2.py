
import json
import numpy as np
from PIL import Image
from Feature import Feature
from collections import Counter

def normalize(feature):
    # chuẩn hóa min max
    min = np.min(feature)
    max = np.max(feature)
    return (feature - min) / (max - min)


# Đọc dữ liệu từ file JSON
with open('data2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Lấy dữ liệu'
feature_data = data['data']


image1 = Image.open('./Flower/hoa anh thảo/hoa anh thảo04.png')
# image1 = Image.open('./Flower/test/hoa hồng.png')
# image1 = Image.open('./test/test/hoa thủy tiên.png')
image1 = np.array(image1)

feature = Feature()
color_feature = feature.color_histogram(image1)
hog_feature = feature.hog(image1)
# lbp_feature = feature.lbp(image1)
# color_feature = color_feature / np.linalg.norm(color_feature)
# hog_feature = hog_feature / np.linalg.norm(hog_feature)

list_distance = []
color_distance = []
hog_distance = []
list_labels = []
for data in feature_data:
    label = data['label']
    color_feature2 = np.array(data['color_feature'])
    hog_feature2 = np.array(data['hog_feature'])
    # lbp_feature2 = np.array(data['lbp_feature'])

    # color_feature2 = color_feature2 / np.linalg.norm(color_feature2)
    # hog_feature2 = hog_feature2 / np.linalg.norm(hog_feature2)

    distance_color = feature.distanceEuclidean(color_feature, color_feature2) * 0.65
    distance_hog = feature.distanceEuclidean(hog_feature, hog_feature2) * 0.35
    # distance_lbp = feature.distanceEuclidean(lbp_feature, lbp_feature2)
    distance =  distance_color  + distance_hog
    
    list_distance.append((label, distance))

    color_distance.append( distance_color)
    hog_distance.append( distance_hog)
    list_labels.append(label)

list_distance.sort(key=lambda x: x[1])
list_distance = list_distance[:5]
print(list_distance)  
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
# Kiểm tra xem nếu có nhiều hơn 1 nhãn xuất hiện cùng số lần, lấy nhãn có giá trị bé nhất
most_common_label = label_count[0][0]

print(most_common_label)

color_distance_normalize = normalize(color_distance)
hog_distance_normalize = normalize(hog_distance)

combine_distance_normalize = color_distance_normalize + hog_distance_normalize

combine_distance = [(list_labels[i], color_distance[i], hog_distance[i], color_distance_normalize[i]) for i in range(len(color_distance))]
combine_distance.sort(key=lambda x: x[3])
for i in range(5):
    print( combine_distance[i])  
# distance = color_distance + hog_distance
# list_distance = list(zip(list_label, distance))

# list_distance.sort(key=lambda x: x[1])
# list_distance = list_distance[:5]
# print(list_distance)
color_distance.sort()
hog_distance.sort()
print(color_distance[-1] - color_distance[1])
print(hog_distance[-1] - hog_distance[1])