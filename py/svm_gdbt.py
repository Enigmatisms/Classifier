# -*- coding: utf-8 -*-
"""
    svm和gdbt算法模块
    @author 谢金豆
"""
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np 
import pandas as pd  
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler #标准差标准化
from sklearn.svm import SVC                      #svm包中SVC用于分类 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
L=["cr","in","pa","ps","rs","sc"]
L1=["cr","in","pa","ps","rs","sc","gg","rp","sp"]
def get_data(x,y):
    file_path='D:/NLS_data/'  #设置文件路径
    file_path1='D:/train/'
    train_set = np.zeros(shape=[1,64*64])  #train_set用于获取的数据集
    train_set = pd.DataFrame(train_set)     #将train_set转换成DataFrame类型
    target=[]                               #标签列表
    for i in L:  
        for j in range(x,y):             
            target.append(i)                
            img = cv2.imread(file_path+i+'/'+str(j)+'.jpg',\
            cv2.IMREAD_GRAYSCALE) #读取图片，第二个参数表示以灰度图像读入
            img=img.reshape(1,img.shape[0]*img.shape[1])
            img=pd.DataFrame(img)           
            train_set=pd.concat([train_set,img],axis=0)
    train_set.index=list(range(0,train_set.shape[0])) 
    train_set.drop(labels=0,axis=0,inplace=True) 
    target=pd.DataFrame(target)            
    return train_set,target                 #返回数据集和标签
def get_data1(x,y):
    file_path='D:/NLS_data/'  #设置文件路径
    file_path1='D:/train/'
    train_set = np.zeros(shape=[1,64*64])  #train_set用于获取的数据集
    train_set = pd.DataFrame(train_set)     #将train_set转换成DataFrame类型
    target=[]                               #标签列表
    for i in range(len(L1)):  
        for j in range(x,y):             
            target.append(i)                
            img = cv2.imread(file_path1+L1[i]+'/'+str(j)+'.jpg',\
            cv2.IMREAD_GRAYSCALE) #读取图片，第二个参数表示以灰度图像读入
            img=img.reshape(1,img.shape[0]*img.shape[1])
            img=pd.DataFrame(img)           
            train_set=pd.concat([train_set,img],axis=0)
    train_set.index=list(range(0,train_set.shape[0])) 
    train_set.drop(labels=0,axis=0,inplace=True) 
    target=pd.DataFrame(target)            
    return train_set,target                 #返回数据集和标签
def AUC_curve(target_test,Y_score):
    Y_Mat=np.zeros(shape=(270,9))
    for i in range(270):
        temp=[0,0,0,0,0,0,0,0,0]
        temp[target_test[i][0]]=1
        Y_Mat[i]=np.array(temp)
    auc1 = metrics.roc_auc_score(Y_Mat, Y_score, average = 'micro')
    fpr, tpr, _ = metrics.roc_curve(Y_Mat.ravel(), Y_score.ravel())
    plt.cla()
    plt.clf()
    plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f'% auc1)
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize = 13)
    plt.ylabel('True Positive Rate', fontsize = 13)
    plt.grid(b = True, ls = ':')
    plt.legend(loc = 'lower right', fancybox = True, framealpha = 0.8, fontsize = 12)
    #plt.title(u'工业缺陷分类后的ROC和AUC', fontsize=17)
    plt.show()

if __name__ == '__main__':
    #1、获取数据
    data_train,target_train=get_data1(1,151)
    #data_test,target_test=get_data(1,181)
    data_test,target_test=get_data1(151,181)
    #2、数据标准化 标准差标准化
    stdScaler = StandardScaler().fit(data_train) 
    trainStd = stdScaler.transform(data_train)
    #stdScaler = StandardScaler().fit(data_test) 
    testStd = stdScaler.transform(data_test)
    #3、建立SVM模型  默认为径向基核函数kernel='rbf' 多项式核函数kernel='poly'
    #clf_svc = GridSearchCV(estimator=SVC(), param_grid={ 'C': [1,3,10,20], 'kernel': [ 'poly','linear', 'rbf'], }, cv=2 ) 

    #svm = clf_svc.fit(trainStd,target_train)
    #print("clf_svc.best_params_   ",clf_svc.best_params_)
    gbm0 = GradientBoostingClassifier(learning_rate=0.20, n_estimators=60, min_samples_leaf=20, 
      max_features='sqrt', subsample=0.8, random_state=40)
    gbm0.fit(trainStd,target_train)
    y_pred = gbm0.predict(testStd)
    # 为每个类别计算ROC曲线和AUC
    Y_score = gbm0.decision_function(testStd)
    target_test=target_test.values    #Dataframe转ndarray方便后面准确率的判断
    AUC_curve(target_test,Y_score)
    print('建立的GDBT模型为：\n',gbm0)
    #print('预测前10个结果为：\n',y_pred[:10])
    true=0
    for i in range(0,270):
        if y_pred[i] == target_test[i]:
            true=true+1
    print('预测结果准确率为：', true/target_test.shape[0])
    Matrix_c=np.zeros(shape=(9,9))
    for i in range(270):
        m=y_pred[i]
        n=target_test[i]
        Matrix_c[m][n]+=1
    sns.heatmap(Matrix_c)
    plt.xlabel('Prediction result')
    plt.ylabel('Ground truth')
    plt.show()
    svm = SVC(kernel='poly',C=3,degree=3).fit(trainStd,target_train)
    Y_score = svm.decision_function(testStd)
    AUC_curve(target_test,Y_score)
    print('建立的SVM模型为：\n',svm)
    #4、预测训练集结果
    target_pred = svm.predict(testStd)
    print('预测前10个结果为：\n',target_pred[:10])
    #target_test=target_test.values    #Dataframe转ndarray方便后面准确率的判断
    true=0
    T1=list(target_pred)
    T2=[]
    for i in range(270):
        T2.append(list(target_test[i]))
    ## 求出预测和真实一样的数目
    #true = np.sum(face_target_pred == face_target_test )
    for i in range(0,270):
        if target_pred[i] == target_test[i]:
            true=true+1
    print('预测结果准确率为：', true/target_test.shape[0])
    Matrix_c1=np.zeros(shape=(9,9))
    for i in range(270):
        m=target_pred[i]
        n=target_test[i]
        Matrix_c1[m][n]+=1
    sns.heatmap(Matrix_c1)
    plt.xlabel('Prediction result')
    plt.ylabel('Ground truth')
    plt.show()
