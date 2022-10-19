import cv2
from cv2 import CV_32F
from cv2 import FONT_HERSHEY_COMPLEX
import hog
import numpy as np
from libsvm.svmutil import *


def detect(img,img_ori,alpha,slide):
    people = []
    m = svm_load_model("model_SVC.model")
    for a in alpha:
        x_true = int(128 * a)
        y_true = int(64 * a)

        for x in range(0,img.shape[0]-x_true,slide):
            for y in range(0,img.shape[1]-y_true,slide):
                temp = img[x:x+x_true,y:y+y_true]
                imgs = cv2.resize(temp,(64,128))
                Hog = hog.Hog_descriptor(imgs, cell_size=8, bin_size=8)
                vector, image = Hog.extract()
                vector = np.array(vector).flatten()
                vec = []
                vec.append(vector)

                p_label, p_acc, p_val = svm_predict([[1]],vec,m)
                if p_label == [1]:
                    print("yes")
                    people.append([y,x,a,p_acc[1]])
                    with open("res.txt","a") as f:
                        f.write("({},{})\n".format(y,x))

    for pos in people:
        img_ori = cv2.rectangle(img_ori, (pos[0],pos[1]), (pos[0]+int(64*pos[2]),pos[1]+int(128*pos[2])), (0, 255, 0), 2)
        img_ori = cv2.putText(img_ori,"err = {}".format(round(pos[3],2)),(pos[0],pos[1]),FONT_HERSHEY_COMPLEX,0.4,(0,0,0),1)

    return img_ori

img_ori = cv2.imread('crop_000027.png')
img = cv2.imread('crop_000027.png', cv2.IMREAD_GRAYSCALE)
# alpha = np.linspace(1,3,4)
alpha = [4]
res = detect(img,img_ori,alpha,20)
cv2.imwrite("2.jpg",res)



# print(img.shape)
# imgs = cv2.resize(img,(64,128))
# # hog = hog.Hog_descriptor(imgs, cell_size=8, bin_size=8)
# # vector, image = hog.extract()
# # vec = np.array(vector).flatten()
# # print(vec.shape)

# img = cv2.rectangle(img, (0,20), (20,40), (0, 255, 0), 2)
# cv2.imwrite("1.jpg",img)