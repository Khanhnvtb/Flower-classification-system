import numpy as np
import math


class Gradient:
    def __init__(self, left=0, right=0, top=0, bottom=0):
        self.Gx = right - left
        self.Gy = top - bottom
        self.total_gradient = math.sqrt(self.Gx * self.Gx + self.Gy * self.Gy)
        angle = np.arctan2(self.Gy, self.Gx)
        self.angle = angle if angle >= 0 else math.pi + angle 
    
    def calculateGradient(image):
        row_size, col_size = image.shape[0], image.shape[1]
        if image.ndim == 2:
            image = image.reshape((row_size, col_size, 1))
            isGrayImage = True
        else:
            isGrayImage = False
        gradient = []
        for row in range(row_size):
            gradient.append([])
            for col in range(col_size):
                gradient[row].append(Gradient())
                row_left = row
                col_left = col - 1
                row_right = row
                col_right = col + 1
                row_top = row - 1
                col_top = col
                row_bottom = row + 1
                col_bottom = col
                if isGrayImage:
                    length = 1
                else:
                    length = 3
                for i in range(length):
                # i = {
                #     0: 'red',
                #     1: 'green',
                #     2: 'blue',
                # }
                    if Gradient.isExist(row_left, col_left, row_size, col_size):
                        left = image[row_left][col_left][i].astype(np.int32)
                    else:
                        left = 0
                    if Gradient.isExist(row_right, col_right, row_size, col_size):
                        right = image[row_right][col_right][i].astype(np.int32)
                    else:
                        right = 0
                    if Gradient.isExist(row_top, col_top, row_size, col_size):
                        top = image[row_top][col_top][i].astype(np.int32)
                    else:
                        top = 0
                    if Gradient.isExist(row_bottom, col_bottom, row_size, col_size):
                        bottom = image[row_bottom][col_bottom][i].astype(np.int32)
                    else:
                        bottom = 0  
                    gradient_by_color = Gradient(left, right, top, bottom)
                    if gradient[row][col].total_gradient < gradient_by_color.total_gradient:
                        gradient[row][col].total_gradient = gradient_by_color.total_gradient
                        gradient[row][col].angle = gradient_by_color.angle
        return np.array(gradient)
    

    def isExist(x, y, row_size, col_size):
        if x < 0 or y == col_size or y < 0 or x == row_size:
            return False
        return True



class Feature:
    def __init__(self):
        self.color = np.empty(0)
        self.shape = np.empty(0)
        self.texture = np.empty(0)

    def __str__(self):
        return f"color : {self.color.shape}\nshape : {self.shape.shape}\ntexture : {self.texture.shape}"

    def rbgToGray(self, image):
        row_size, col_size = image.shape[0], image.shape[1]
        gray_image = np.empty ((row_size, col_size))
        for row in range(row_size):
            for col in range(col_size):
                # {
                #     0: 'red',
                #     1: 'green',
                #     2: 'blue',
                # }
                # Grayscale = 0.3*R+0.59*G+ 0.11*B
                gray_image[row][col] = int(image[row][col][0] * 0.3 + image[row][col][1] * 0.59 + image[row][col][2] * 0.11)
        return gray_image


    def calculateFeature(self, histogram, cells_per_block):
        vector_features = []
        row_block_size = histogram.shape[0]
        col_block_size = histogram.shape[1]
        row_cell_start, col_cell_start, row_cell_end, col_cell_end = 0, 0, cells_per_block[0], cells_per_block[1]
        while(True):
            block = histogram[row_cell_start:row_cell_end, col_cell_start:col_cell_end]
            k = math.sqrt(np.sum(block * block))
            if k != 0:
                block = block / k
            vector_features.append(block)
            if col_cell_end < col_block_size:
                col_cell_start += 1
                col_cell_end += 1
            else:
                if row_cell_end < row_block_size:
                    row_cell_start += 1
                    row_cell_end += 1
                    col_cell_start = 0
                    col_cell_end = cells_per_block[1]
                else:
                    break
        vector_features = np.array(vector_features)
        return vector_features.reshape(-1)


    def calculateHistogramOfGradient(self, gradient, orientations, pixel_per_cell):
        row_size, col_size = gradient.shape[0], gradient.shape[1]
        histogram = []
        histogram_of_row = []
        row_start, col_start, row_end, col_end = 0, 0, pixel_per_cell[0], pixel_per_cell[1]
        pi_per_orientations = math.pi / orientations
        while(True):
            histogram_of_cell = [0 for i in range(orientations)]
            cell = gradient[row_start:row_end, col_start:col_end]
            # chuyển thành 1 chiều
            cell = np.ravel(cell)
            # chuyển thành mảng chứa 1 mảng total_gradient và 1 mảng chứa angle
            vfunc = np.vectorize(self.getValue)
            cell = vfunc(cell)
            for i in range(len(cell[0])):
                total_gradient = cell[0][i]
                angle = cell[1][i]
                angle_per_pi_per_orientations = angle / pi_per_orientations
                if angle_per_pi_per_orientations == int(angle_per_pi_per_orientations):
                    left_bin = int(angle_per_pi_per_orientations) - 1
                else:
                    left_bin = int(angle_per_pi_per_orientations)
                left_angle = left_bin * pi_per_orientations
                right_angle = (left_bin + 1) * pi_per_orientations
                if left_bin < orientations - 1:
                    right_bin = left_bin + 1
                else:
                    right_bin = 0
                histogram_of_cell[left_bin] += (right_angle - angle) / pi_per_orientations * total_gradient
                histogram_of_cell[right_bin] += (angle - left_angle) / pi_per_orientations * total_gradient
            histogram_of_row.append(histogram_of_cell)
            if col_end < col_size:
                col_start = col_end
                col_end = col_start + pixel_per_cell[1]
            else:
                histogram.append(histogram_of_row)
                histogram_of_row = []
                if row_end < row_size:
                    row_start = row_end
                    row_end = row_start + pixel_per_cell[0]
                    col_start = 0
                    col_end = pixel_per_cell[1]
                else:
                    break
        return np.array(histogram)

    def calculateHistogramOfLbp(self, lbp_values, orientations, pixel_per_cell):
        row_size, col_size = lbp_values.shape[0], lbp_values.shape[1]
        histogram = []
        histogram_of_row = []
        row_start, col_start, row_end, col_end = 0, 0, pixel_per_cell[0], pixel_per_cell[1]
        max_lbp_value_per_orientations = 256 / orientations
        while(True):
            histogram_of_cell = [0 for i in range(orientations)]
            cell = lbp_values[row_start:row_end, col_start:col_end]
            # chuyển thành 1 chiều
            cell = np.ravel(cell)
            for lbp_value in cell:
                lbp_value_per_max_lbp_value_per_orientations = lbp_value / max_lbp_value_per_orientations
                if lbp_value_per_max_lbp_value_per_orientations == int(lbp_value_per_max_lbp_value_per_orientations):
                    left_bin = int(lbp_value_per_max_lbp_value_per_orientations) - 1
                else:
                    left_bin = int(lbp_value_per_max_lbp_value_per_orientations)
                left_lbp_value = left_bin * max_lbp_value_per_orientations
                right_lbp_value = (left_bin + 1) * max_lbp_value_per_orientations
                if left_bin < orientations - 1:
                    right_bin = left_bin + 1
                else:
                    right_bin = 0
                histogram_of_cell[left_bin] += right_lbp_value - lbp_value
                histogram_of_cell[right_bin] += lbp_value - left_lbp_value
            histogram_of_row.append(histogram_of_cell)
            if col_end < col_size:
                col_start = col_end
                col_end = col_start + pixel_per_cell[1]
            else:
                histogram.append(histogram_of_row)
                histogram_of_row = []
                if row_end < row_size:
                    row_start = row_end
                    row_end = row_start + pixel_per_cell[0]
                    col_start = 0
                    col_end = pixel_per_cell[1]
                else:
                    break
        return np.array(histogram)

    def calculateLbp(self, gray_image):
        d_row = [0, -1, -1, -1, 0, 1, 1, 1]
        d_col = [1, 1, 0, -1, -1, -1, 0, 1]
        row_size, col_size = gray_image.shape[0], gray_image.shape[1]
        lbp_values = np.empty((row_size, col_size), dtype=np.ubyte)
        for row in range(row_size):
            for col in range(col_size):
                for i in range(8):
                    lbp_value = 0
                    curr_row = row + d_row[i]
                    curr_col = col + d_col[i]
                    if Gradient.isExist(curr_row, curr_col, row_size, col_size):
                        lbp_value += np.ubyte(math.pow(2, i)) if gray_image[curr_row][curr_col] > gray_image[row][col] else np.ubyte(0)
                    else:
                        lbp_value += np.ubyte(0)
                lbp_values[row][col] = lbp_value
        return lbp_values
    
    def getValue(self, g):
        return (g.total_gradient, g.angle) 

    def hog(self, image, orientations=9, pixel_per_cell=(8,8), cells_per_block=(2,2)):
        gray_image = self.rbgToGray(image)
        gradient = Gradient.calculateGradient(gray_image)
        histogram = self.calculateHistogramOfGradient(gradient, orientations, pixel_per_cell)
        self.shape = self.calculateFeature(histogram, cells_per_block)
        return self.shape

    def lbp(self, image, orientations=8, pixel_per_cell=(8,8), cells_per_block=(3,3)):
        gray_image = self.rbgToGray(image)
        lbp_values = self.calculateLbp(gray_image)
        histogram = self.calculateHistogramOfLbp(lbp_values, orientations, pixel_per_cell)
        self.texture = self.calculateFeature(histogram, cells_per_block)
        return self.texture

    def color_histogram(self, image, num_bins=16, block=(16,16)):
        vector_features = []
        num_block_row = int(image.shape[0] / block[0])
        num_block_col = int(image.shape[1] / block[1])

        for i in range(num_block_row):
            for j in range(num_block_col):
                block_image = image[i*block[0]: (i+1)*block[0], j*block[1]: (j+1)*block[1]]
                histogram = self.calculateRGBHistogram(block_image, num_bins)
                vector_features.append(histogram)
        vector_features = np.concatenate(vector_features)
        return vector_features
    
    def calculateRGBHistogram(self, image, num_bins):
        value_bin = 256 / num_bins
        histogram_of_red    = np.zeros(num_bins)
        histogram_of_green  = np.zeros(num_bins)
        histogram_of_blue   = np.zeros(num_bins)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram_of_red[int(image[i][j][0] / 256 * value_bin)] += 1
                histogram_of_green[int(image[i][j][1] / 256 * value_bin)] += 1
                histogram_of_blue[int(image[i][j][2] / 256 * value_bin)] += 1
        
        histogram = np.concatenate((histogram_of_red, histogram_of_green, histogram_of_blue))
        # histogram = histogram / np.sum(histogram)
        histogram = histogram / np.linalg.norm(histogram)
        
        return histogram
    
    def calculateCombinedRGBHistogram(self, image, num_bins):
        histogram = np.zeros(num_bins * num_bins * num_bins)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                index_of_red = int(image[i][j][0] / 256 * num_bins)
                index_of_green = int(image[i][j][1] / 256 * num_bins)
                index_of_blue = int(image[i][j][2] / 256 * num_bins)
                histogram[index_of_red * num_bins * num_bins + index_of_green * num_bins + index_of_blue] += 1
        
        histogram = histogram / np.linalg.norm(histogram)
        return histogram


    def distanceEuclidean(self, feature1, feature2):
        distance = 0
        for i in range(len(feature1)):
            distance += (feature1[i] - feature2[i]) * (feature1[i] - feature2[i])
        distance = math.sqrt(distance)
        return distance