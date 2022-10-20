# 基于HOG特征和SVM、LR模型实现行人检测

## 实验目的

​		使用SVM和Logistic Regression模型实现行人检测功能

## 实验步骤

​		1，提取INRIAPerson训练集的HOG特征

​		2，基于提取的HOG特征进行训练

​		3，在INRIAPerson验证集上进行验证

​		4，绘制ROC曲线，计算AUC

​		5，基于已有模型进行行人检测

## 实验设计

### 训练集再分类

​		初始的INRIAPerson训练集十分杂乱，其不规整性和指针链接非常不利于实验的进行，故在进行实验之前，首先是对训练集进行整理分类，得到train和test文件夹，分别存放训练集和数据集，便于实验顺利进行。其中，正样本均为经过padding的原大小为64x128大小的人类图像，负样本均为大小不规则的不包含人类的图片。

### HOG特征的提取

​		基于HOG提取方法：把样本图像分割为若干个像素的单元cell，把梯度方向平均划分为bin_size个区间，在每个cell里面对所有像素的梯度方向在各个方向区间进行直方图统计，得到特征向量，每相邻的4个cell构成一个block，把一个block内的特征向量联起来得到一个block特征向量，用块对样本图像进行扫描，扫描步长为一个cell。最后将所有块的特征串联起来，就得到了人体的特征。

​		以此处为例，基于INRIAPerson训练集用例64x128的图像大小，总共能提取出32x7x15=3360个特征。

​		具体的提取方式是：对于正样本，消除padding影响后直接提取HOG特征，对于负样本，取十个大小为64x128的像素块提取HOG特征。

```python
#HOG特征提取接口
#img_temp：输入图像
#cell_size：cell尺度
#bin_size：按角分块数目 
hog = Hog_descriptor(img_temp, cell_size=8, bin_size=8)
```

### 模型训练

#### 基于LIBSVM

```python
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
```

#### 基于LogisticRegression

```python
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
```

### 模型验证

#### 基于LIBSVM

```python
def test(type): 
    pvec_t , nvec_t = hog.load_train()
    x_test , y_test = [] , []
    for i in pvec_t:
        y_test.append(1)
        x_test.append(i)

    for i in nvec_t:
        y_test.append(0)
        x_test.append(i)

    m = svm_load_model("model_{}.model".format(type))
```

#### 基于LogisticRegression

```python
#代码承接上述train_logister
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
```

### ROC绘制和AUC计算

```python
fpr, tpr, thersholds = roc_curve(y_test, p_label, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
```

### 行人检测

​		对任意一张图片，确定尺度比例alpha，构建大小为（64*alpha，128\*alpha）的搜索框，以slide_size大小的滑动步长进行检索。需要注意，每一个检索框都应被归一化到64x128大小，以产生相同维度的HOG特征。

​		对于SVM模型，因为其二分类的功能，结果具有“全或无”的特性，只要被检测出为行人，所有的搜索框具有相同的置信度，故展示所有被检索为人的结果。而对于LR模型，则将结果进行筛选：只有当置信度大于一个值（选定为0.99）时，才被判定为有效结果。

```
#行人检测接口
#img：灰度图
#img_ori：彩色原图
#alpha：尺度比例
#slide_size：滑动步长
res = detect(img,img_ori,alpha,slide_size)
```

## 遇上的困难及解决方式

​		1，图像可能无法读取：因为数据集里面不是图片而是快捷方式

​		2，图像尺寸不一致：数据集实现对图像进行padding导致

​		3，训练集分类杂乱：重新分类

## 优化点

​		可以统一HOG特征输出格式，不必建立在.lst文件基础上进行HOG特征读取，增加美观性和可读性。

​		可以进一步优化接口，使各个接口格式统一
