import numpy as np
import math


def isExist(x, y, row_size, col_size):
    if x < 0 or y == col_size or y < 0 or x == row_size:
        return False
    return True


class Gradient:
    def __init__(self, left=0, right=0, top=0, bottom=0):
        self.Gx = right - left
        self.Gy = top - bottom
        self.total_gradient = math.sqrt(self.Gx * self.Gx + self.Gy * self.Gy)
        self.angle = np.arctan2(self.Gy, self.Gx)
    
    def calculateGradient(image, row_size, col_size):
        gradient = []
        for row in range(row_size):
            gradient.append([])
            for col in range(col_size):
                gradient[row].append(Gradient())
                x_left = row
                y_left = col - 1
                x_right = row
                y_right = col + 1
                x_top = row - 1
                y_top = col
                x_bottom = row + 1
                y_bottom = col
                for i in range(3):
                # i = {
                #     0: 'red',
                #     1: 'green',
                #     2: 'blue',
                # }
                    if isExist(x_left, y_left, row_size, col_size):
                        left = image[x_left][y_left][i].astype(np.int32)
                    else:
                        left = 0
                    if isExist(x_right, y_right, row_size, col_size):
                        right = image[x_right][y_right][i].astype(np.int32)
                    else:
                        right = 0
                    if isExist(x_top, y_top, row_size, col_size):
                        top = image[x_top][y_top][i].astype(np.int32)
                    else:
                        top = 0
                    if isExist(x_bottom, y_bottom, row_size, col_size):
                        bottom = image[x_bottom][y_bottom][i].astype(np.int32)
                    else:
                        bottom = 0  
                    gradient_by_color = Gradient(left, right, top, bottom)
                    if gradient[row][col].total_gradient < gradient_by_color.total_gradient:
                        gradient[row][col].total_gradient = gradient_by_color.total_gradient
                        gradient[row][col].angle = gradient_by_color.angle
        return np.array(gradient)
    

class Feature:
    def __init__(self):
        self.color = []
        self.shape = []
        self.structure = []

    def getValue(self, g):
        return (g.total_gradient, g.angle)

    def calculateHistogram(self, gradient, orientations, pixel_per_cell, row_size, col_size):
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
            pi_per_orientations = math.pi / 8
            for i in range(len(cell[0])):
                total_gradient = cell[0][i]
                angle = cell[1][i]
                angle_per_pi_per_orientations = angle / pi_per_orientations
                left_bin = int(angle_per_pi_per_orientations)
                left_angle = left_bin * pi_per_orientations
                right_angle = (left_bin + 1) * pi_per_orientations
                if angle_per_pi_per_orientations < orientations - 1:
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

    def hog(self, image, orientations=9, pixel_per_cell=(8,8), cells_per_block=(3,3)):
        row_size, col_size = image.shape[0], image.shape[1]
        gradient = Gradient.calculateGradient(image, row_size, col_size)
        histogram = self.calculateHistogram(gradient, orientations, pixel_per_cell, row_size, col_size)
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