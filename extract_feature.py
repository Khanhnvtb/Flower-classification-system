from Feature import Feature
from PIL import Image
import numpy as np
import json

import os
folder_path = './Flower/'
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

feature_data = []
feature = Feature()
for subfolder in subfolders:
    # Lấy danh sách các tệp trong thư mục
    files = os.listdir(os.path.join(folder_path, subfolder))
    
    for file_name in files:
        data = {}
        data['label'] = subfolder
        # Đọc ảnh
        img = Image.open(os.path.join(folder_path, subfolder, file_name))
        img = np.array(img)
        # Tính toán đặc trưng
        color_feature = feature.color_histogram(img)
        hog_feature = feature.hog(img)
        lbp_feature = feature.lbp(img)
        data['color_feature'] = color_feature.tolist()
        data['hog_feature'] = hog_feature.tolist()
        feature_data.append(data)

data = {}
data['data'] = feature_data

# Lưu biến feature_data ra file JSON
with open('data3.json', 'w', encoding='utf-8') as f:
    json.dump(data, f)