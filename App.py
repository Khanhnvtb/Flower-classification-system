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

        self.open_button = tk.Button(self, text="Mở ảnh", command=self.open_file)
        self.open_button.pack(pady=15, ipadx=10,)
        self.label = tk.Label(self)
        self.label.pack(pady=10)
        
        self.image_input = tk.Label(self)
        self.image_input.pack()

        self.result = tk.Label(self)
        self.result.pack(pady=15)
        
        self.feature = Feature()
        self.data = None
        self.load_data()

    def load_data(self):
        with open('data.json', 'r') as f:
            data = json.load(f)
            self.data = data['data']

            
    def click_me(self):
        self.image_input.configure(text="I have been clicked!")

    def open_file(self):
        file = filedialog.askopenfile(initialdir=os.path.abspath(os.getcwd()), title="Select file", filetypes=(("png files", "*.png"), ("all files", "*.*")))
        if file:
            print(file.name)
            self.label.configure(text="Hình ảnh đầu vào")
            self.image = tk.PhotoImage(file=file.name)
            self.image_input.config(image=self.image)

            self.predict(file.name)

    def predict(self, image_file):
        image = Image.open(image_file).resize((128, 128))
        image = np.array(image)

        color_feature_of_test = self.feature.color_histogram(image)
        hog_feature_of_test = self.feature.hog(image)

        list_distance = []

        for data in self.data:
            label = data['label']
            color_feature_of_train = np.array(data['color_feature'])
            hog_feature_of_train = np.array(data['hog_feature'])
            
            distance_color = self.feature.distanceEuclidean(color_feature_of_test, color_feature_of_train)
            distance_hog = self.feature.distanceEuclidean(hog_feature_of_test, hog_feature_of_train)
            distance = distance_color + distance_hog
            list_distance.append((label, distance))


        list_distance.sort(key=lambda x: x[1])
        list_distance = list_distance[:5]
        list_label = [x[0] for x in list_distance]
        # get element frequency max in list
        label = max(set(list_label), key=list_label.count)
        
        self.result.configure(text="Kết quả dự đoán: " + label)
        
if __name__ == "__main__":
    app = App()
    app.mainloop()