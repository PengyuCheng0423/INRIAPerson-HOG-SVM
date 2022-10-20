from sklearn.linear_model import LogisticRegression
import hog
import cv2
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np

def train_logister(img,img_ori,alpha,slide):
    pvec , nvec = hog.load_hog()
    x , y = [] , []
    for i in pvec:
        y.append(1)
        x.append(i)

    for i in nvec:
        y.append(0)
        x.append(i)

    pvec_t , nvec_t = hog.load_train()
    x_test , y_test = [] , []
    for i in pvec_t:
        y_test.append(1)
        x_test.append(i)

    for i in nvec_t:
        y_test.append(0)
        x_test.append(i)
    
    model_1 = LogisticRegression(max_iter=10000)

    model_1.fit(x,y)

    s = model_1.score(x_test,y_test)
    print("ACC = {}".format(s))

    people = []
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
                
                p_label = 2
                acc = model_1.predict_proba(vec)
                # print(acc)
                if acc[0][1] >= 0.99:
                    p_label = 1
                else:
                    p_label = 0

                if p_label == 1:
                    # print("yes")
                    people.append([y,x,a,acc[0][1]])
                    with open("res.txt","a") as f:
                        f.write("({},{})\n".format(y,x))

    for pos in people:
        img_ori = cv2.rectangle(img_ori, (pos[0],pos[1]), (pos[0]+int(64*pos[2]),pos[1]+int(128*pos[2])), (0, 255, 0), 2)
        img_ori = cv2.putText(img_ori,"acc = {}".format(round(pos[3],2)),(pos[0],pos[1]),FONT_HERSHEY_COMPLEX,0.4,(0,0,0),1)

    return img_ori

img_ori = cv2.imread('person_138.png')
img = cv2.imread('person_138.png', cv2.IMREAD_GRAYSCALE)
# alpha = np.linspace(1,4,6)
alpha = [3]
res = train_logister(img,img_ori,alpha,10)
cv2.imwrite("slide10.jpg",res)
