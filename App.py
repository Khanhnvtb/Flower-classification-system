import tkinter as tk
from tkinter import filedialog
import os
import json 
import numpy as np
from PIL import Image
from Feature import Feature

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phân loại hoa")
        self.width = 500
        self.height = 500
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width / 2) - (self.width / 2)
        y = (screen_height / 2) - (self.height / 2)
        self.geometry('{}x{}+{}+{}'.format(self.width, self.height, int(x), int(y)))

        self.open_button = tk.Button(self, text="Chọn ảnh", command=self.open_file)
        self.open_button.pack(pady=15, ipadx=10,)
        self.label = tk.Label(self)
        self.label.pack(pady=10)
        
        self.image_input = tk.Label(self,)
        self.image_input.pack()

        self.result = tk.Label(self)
        self.result.pack(pady=15)
        
        self.feature = Feature()
        self.data = None
        self.load_data()

    def load_data(self):
        with open('data.json', 'r',encoding='utf-8') as f:
            data = json.load(f)
            self.data = data['data']

            
    def click_me(self):
        self.image_input.configure(text="I have been clicked!")

    def open_file(self):
        file = filedialog.askopenfile(initialdir=os.path.abspath(os.getcwd() + '/Flower/test'), title="Select file", filetypes=(("png files", "*.png"), ("jpg files", "*.jpg")))
        if file:
            print(file.name)
            self.label.configure(text="Hình ảnh đầu vào")
            self.image = tk.PhotoImage(file=file.name)
            self.image_input.config(image=self.image)

            self.predict(file.name)

    def predict(self, image_file):
        image = Image.open(image_file)
        image = np.array(image)

        color_feature = self.feature.color_histogram(image)
        hog_feature = self.feature.hog(image)

        list_distance = []
        list_color_distance = []
        list_hog_distance = []
        for data in self.data:
            label = data['label']
            color_feature2 = np.array(data['color_feature'])
            hog_feature2 = np.array(data['hog_feature'])
            
            distance_color = self.feature.distanceEuclidean(color_feature, color_feature2)
            distance_hog = self.feature.distanceEuclidean(hog_feature, hog_feature2)
            distance = distance_color + distance_hog
            list_distance.append((label, distance))
            
            list_color_distance.append(distance_color)
            list_hog_distance.append(distance_hog)
            

        list_distance.sort(key=lambda x: x[1])



        if list_distance[5][1] > 21:
            self.result.configure(text="Không xác định được loại hoa")
            return

        list_distance = list_distance[:5]
        label_count = {}
        for label, distance in list_distance:
            if label in label_count:
                label_count[label]['count'] += 1
                if distance < label_count[label]['distance']:
                    label_count[label]['distance'] = distance
            else:
                label_count[label] = {
                    'count': 1,
                    'distance': distance
                }
        
        label_count = sorted(label_count.items(), key=lambda x: (-x[1]['count'],x[1]['distance']))

        most_common_label = label_count[0][0]
        
        # list_color_distance.sort()
        # list_hog_distance.sort()
        # print(list_color_distance[-1] - list_color_distance[0], list_color_distance[-1], list_color_distance[0])
        # print(list_hog_distance[-1] - list_hog_distance[0], list_hog_distance[-1], list_hog_distance[0])
        
        self.result.configure(text="Kết quả dự đoán: " + most_common_label)
        

if __name__ == "__main__":
    app = App()
    app.mainloop()