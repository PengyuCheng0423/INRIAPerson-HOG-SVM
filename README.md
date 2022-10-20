# INRIAPerson-HOG-SVM
Pedestrian detection is realized based on HOG features and SVM and LR models

My first git :-)

# 使用方法
HOG提取过程略

## LIBSVM
##### 模型训练/验证
命令行调用svm.py，先调用"svm -b train"进行训练，然后调用"svm -b test"在验证集上验证结果

##### 行人检测
在detect_svm中指定待检测图像路径，img为灰度图，img_ori为原图，手动调alpha和slidesize大小
```python
res = detect(img,img_ori,alpha = [3],slide = 20)
```

## LR
一体式封装，train_logister内封装有模型训练、验证、行人检测等所有过程，参数意义同上
```python
res = train_logister(img,img_ori,alpha = [3],slide = 10)
```

