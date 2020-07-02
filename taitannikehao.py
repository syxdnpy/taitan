#加载数据包
import pandas as pd
import numpy as np
import random as rnd


import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from  sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import recall_score


#读取文件
train = pd.read_csv('C:/Users/zht/Desktop/作业4/train.csv')
test = pd.read_csv('C:/Users/zht/Desktop/作业4/test.csv')

combine = [train,test]

#数据预览
print(train.head(5))
print(test.head(5))

#年龄与生还
g = sns.FacetGrid(train,col='Survived')
g.map(plt.hist,'Age',bins=20)

#补充年龄空值
train['Age']=train['Age'].fillna(train['Age'].median())
test['Age']=test['Age'].fillna(test['Age'].median())

#数据描述
print(train.describe())

#船舱等级与生还
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()\
.sort_values(by='Survived',ascending=False)

#性别与生还
train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()\
.sort_values(by='Survived',ascending=False)

#数据归一
if __name__ == '__main__':
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)
print(train.head())

#删除多余参数
train= train.drop(['Ticket','Cabin'],axis=1)
test=test.drop(['Ticket','Cabin'],axis=1)
combine=[train,test]
train=train.drop(['Name','PassengerId'],axis=1)
test=test.drop(['Name'],axis=1)
combine=[train,test]
print(train.shape,test.shape)
train=train.drop(['Parch','SibSp'],axis=1)
test=test.drop(['Parch','SibSp'],axis=1)
combine=[train,test]
print(train.head())

#船舱等级性别与生还
grid = sns.FacetGrid(train,row='Pclass',col='Sex',size=2.2,aspect=1.6)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend()
plt.show()
#港口与生还
freq_port = train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()\
.sort_values(by='Survived',ascending=False)

#港口数据归一
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
train.head()

#划定费用范围
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()\
      .sort_values(by='FareBand', ascending=True)

#费用归一
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train= train.drop(['FareBand'],axis=1)
combine = [train,test]
print(train.head(5))

#建模参数准备，划分训练测试集
features=['Sex','Age','Pclass','Embarked','Fare']
X=train[features]
Y=train['Survived']
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.3, random_state=0)
X_test = test.drop('PassengerId',axis=1).copy()

#CART决策树
clf = DecisionTreeClassifier(criterion='gini')
clf = clf.fit(train_features, train_labels)
test_predict = clf.predict(test_features)
train_predict= clf.predict(train_features)
Y_predict= clf.predict(X_test)
acc_cart = round(accuracy_score(train_labels, train_predict)* 100,2)
print(acc_cart)
score_cart = round(accuracy_score(test_labels, test_predict)* 100,2)
recall_cart= round(recall_score(test_labels, test_predict)* 100,2)
print(acc_cart)
print(recall_cart)

#支持向量机
svc = SVC()
svc.fit(train_features, train_labels)
test_predict = svc.predict(test_features)
acc_svc = round(svc.score(train_features, train_labels) * 100,2)
print(acc_svc)
acc_svc  = round(accuracy_score(test_labels, test_predict)* 100,2)
recall_svc= round(recall_score(test_labels, test_predict)* 100,2)
print(acc_svc)
print(recall_svc)

#朴素贝叶斯
gaussian = GaussianNB()
gaussian.fit(train_features, train_labels)
test_predict= gaussian.predict(test_features)
acc_gaussian = round(gaussian.score(train_features, train_labels)*100,2)
print(acc_gaussian)
acc_gaussian=round(accuracy_score(test_labels, test_predict)* 100,2)
recall_gaussian= round(recall_score(test_labels, test_predict)* 100,2)
print(acc_gaussian)
print(recall_gaussian)

#随机森林
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_features, train_labels)
test_predict = random_forest.predict(test_features)
acc_random_forest = round(random_forest.score(train_features, train_labels)*100,2)
print(acc_random_forest)
acc_random_forest=round(accuracy_score(test_labels, test_predict)* 100,2)
recall_random_forest= round(recall_score(test_labels, test_predict)* 100,2)
Y_predict= random_forest.predict(X_test)
print(acc_random_forest)
print(recall_random_forest)

#计算auc值
fpr, tpr, thresholds = roc_curve(test_labels, test_predict);
roc_auc = auc(fpr, tpr)

#画ROC曲线
lw = 2
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#将预测数据保存入DATAFRAME
test['Survived']= pd.DataFrame(Y_predict)
test['Survived']
print(test)
#保存成CSV
#test.to_csv(path_or_buf="test4.csv",index=False)


