from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import math

def calculate_intersection(hI, hM):
    intersection = 0
    for i in range(len(hI)):
        intersection += (hI[i] - hM[i]) * (hI[i] - hM[i])
    return math.sqrt(intersection)

index = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

fds = {}
fds['hibiscus - hoa dâm bụt'] = []
fds['poppy - hoa anh túc'] = []
fds['primrose - hoa anh thảo'] = []
# fds['peony - hoa mẫu đơn'] = []
fds['pansy - hoa bướm'] = []
fds['helianthemum syriacum - hoa hồng đá'] = []

for i in fds.keys():
    name = i.split(' -')[0]
    for j in index:
        img = imread(f'Flower\{i}\{name}{j}.png')
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
        fds[f'{i}'].append(fd)

img_test = imread('Flower/test/sunflower.png')
img_test = resize(img_test, (128,128))
fd, hog_image_test = hog(img_test, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)

def kNearestNeighbor(trainSet, point, k):
    distances = []
    for i in fds.keys():
        name = i.split(' -')[0]
        for j in fds[i]:
            distance = calculate_intersection(j, fd)
            distances.append((distance, name))
    