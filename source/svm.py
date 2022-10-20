import hog
from libsvm.svmutil import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse

def train(type , s):
    pvec , nvec = hog.load_hog()
    x , y = [] , []
    for i in pvec:
        y.append(1)
        x.append(i)

    for i in nvec:
        y.append(0)
        x.append(i)

    prob = svm_problem(y,x)
    param = svm_parameter("-s {}".format(s))

    m = svm_train(prob,param)

    svm_save_model("model_{}.model".format(type),m)


def test(type): 
# m = m = svm_load_model("")
    pvec_t , nvec_t = hog.load_train()
    x_test , y_test = [] , []
    for i in pvec_t:
        y_test.append(1)
        x_test.append(i)

    for i in nvec_t:
        y_test.append(0)
        x_test.append(i)

    m = svm_load_model("model_{}.model".format(type))

    p_label, p_acc, p_val = svm_predict(y_test, x_test, m, '-b 0')
    ACC, MSE, SCC = evaluations(y_test, p_label)
    print("ACC = {}".format(ACC))
    print("MSE = {}".format(MSE))
    print("SCC = {}".format(SCC))

    fpr, tpr, thersholds = roc_curve(y_test, p_label, pos_label=1)
 
    roc_auc = auc(fpr, tpr)
 
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
 
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Type of the model")
    parser.add_argument('-b','--bool',default= "test")
    parser.add_argument('-t','--type', default= 'SVC')
    parser.add_argument('-s','--stype',default= 0)
    args = parser.parse_args()
    t , s , b = args.type , args.stype ,args.bool
    if b == "train":
        train(t , s)
    elif b == "test":
        test(t)