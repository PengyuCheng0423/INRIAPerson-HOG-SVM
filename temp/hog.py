import random
from turtle import pos
import cv2
import numpy as np
import math
from PIL import Image


class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 // self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"



    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size // 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

def pos_hog():
    f = open("train/pos.lst","r")
    pos_lst=f.readlines()

    for path in pos_lst:
        name = path[10:-5]
        img = Image.open("{}".format(path[:-1]))
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        #消除padding影响
        img_gray = img_gray[16:-16,16:-16]
        hog = Hog_descriptor(img_gray, cell_size=8, bin_size=8)
        vector, image = hog.extract()

        with open("train/poshog/{}.txt".format(name),"w") as f:
            for i in range(len(vector)):
                for j in range(len(vector[0])):
                    f.write(str(vector[i][j])+" ")

def pos_hog_test():
    f = open("test/pos.lst","r")
    pos_lst=f.readlines()

    for path in pos_lst:
        name = path[9:-5]
        img = Image.open("{}".format(path[:-1]))
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        #消除padding影响
        img_gray = img_gray[3:-3,3:-3]
        hog = Hog_descriptor(img_gray, cell_size=8, bin_size=8)
        vector, image = hog.extract()

        with open("test/poshog/{}.txt".format(name),"w") as f:
            for i in range(len(vector)):
                for j in range(len(vector[0])):
                    f.write(str(vector[i][j])+" ")

def neg_hog():
    f = open("train/neg.lst","r")
    pos_lst=f.readlines()

    for path in pos_lst:
        name = path[10:-5]
        img = Image.open("{}".format(path[:-1]))
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        #随机取10个区域
        for times in range(10):
            x_rand = random.randint(0,img_gray.shape[0]-129)
            y_rand = random.randint(0,img_gray.shape[1]-64)
            img_temp = img_gray[x_rand:x_rand+128,y_rand:y_rand+64]
            cv2.imwrite("1.jpg",img_temp)
            if np.min(img_temp) >= 245 or np.max(img_temp) <= 10:
                continue
            hog = Hog_descriptor(img_temp, cell_size=8, bin_size=8)
            vector, image = hog.extract()

            with open("train/neghog.txt","a") as f:
                f.write("train/neghog/{}_{}.txt\n".format(name,times))

            with open("train/neghog/{}_{}.txt".format(name,times),"w") as f:
                for i in range(len(vector)):
                    for j in range(len(vector[0])):
                        f.write(str(vector[i][j])+" ")

def neg_hog_test():
    f = open("test/neg.lst","r")
    pos_lst=f.readlines()

    for path in pos_lst:
        name = path[9:-5]
        img = Image.open("{}".format(path[:-1]))
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        #随机取10个区域
        for times in range(10):
            x_rand = random.randint(0,img_gray.shape[0]-129)
            y_rand = random.randint(0,img_gray.shape[1]-64)
            img_temp = img_gray[x_rand:x_rand+128,y_rand:y_rand+64]
            cv2.imwrite("1.jpg",img_temp)
            if np.min(img_temp) >= 245 or np.max(img_temp) <= 10:
                continue
            hog = Hog_descriptor(img_temp, cell_size=8, bin_size=8)
            vector, image = hog.extract()

            with open("test/neghog.txt","a") as f:
                f.write("test/neghog/{}_{}.txt\n".format(name,times))

            with open("test/neghog/{}_{}.txt".format(name,times),"w") as f:
                for i in range(len(vector)):
                    for j in range(len(vector[0])):
                        f.write(str(vector[i][j])+" ")

def load_hog():
    f_pos = open("train/pos.lst","r")
    f_neg = open("train/neghog.txt","r")

    pos_lst=f_pos.readlines()
    neg_lst=f_neg.readlines()

    pos_vec = []
    neg_vec = []

    for path in pos_lst:
        name = path[10:-5]
        pos_vec.append(np.loadtxt("train/poshog/{}.txt".format(name)))

    for path in neg_lst:
        neg_vec.append(np.loadtxt(path[:-1]))

    return pos_vec,neg_vec

def load_train():
    f_pos = open("test/pos.lst","r")
    f_neg = open("test/neghog.txt","r")

    pos_lst=f_pos.readlines()
    neg_lst=f_neg.readlines()

    pos_vec = []
    neg_vec = []

    for path in pos_lst:
        name = path[9:-5]
        pos_vec.append(np.loadtxt("test/poshog/{}.txt".format(name)))

    for path in neg_lst:
        neg_vec.append(np.loadtxt(path[:-1]))

    return pos_vec,neg_vec



if __name__ == "__main__":
    pos_hog()  
    neg_hog()
    pos_hog_test()
    neg_hog_test()
    # a = np.loadtxt("test\\neghog\\01-03f_0.txt")
    # print(a.shape)
