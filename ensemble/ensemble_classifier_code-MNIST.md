# 앙상블 classifier-MNIST dataset


```python
# import library
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


```python
# load data
mnist=datasets.load_digits()
data, target=mnist.data, mnist.target
x_train, x_test, y_train, y_test=train_test_split(data, target, test_size=0.3)
```

single model accuracy


```python
# decision tree
dt=tree.DecisionTreeClassifier(criterion='gini')
dt=dt.fit(x_train, y_train)
dt_pred=dt.predict(x_test)
```


```python
# KNN
knn=KNeighborsClassifier(n_neighbors=100)
knn=knn.fit(x_train, y_train)
knn_pred=knn.predict(x_test)
```


```python
# SVM
svm=SVC(probability=True)
svm=svm.fit(x_train, y_train)
svm_pred=svm.predict(x_test)
```


```python
# hard voting
voting_clf=VotingClassifier(estimators=[('decision_tree', dt), ('KNN', knn), ('SVM', svm)],
                            weights=[1, 1, 1], voting='hard')
voting_clf=voting_clf.fit(x_train, y_train)
voting_hard_pred=voting_clf.predict(x_test)
```


```python
# soft voting
voting_s_clf=VotingClassifier(estimators=[('decision_tree', dt), ('KNN', knn), ('SVM', svm)],
                              weights=[1, 1, 1], voting='soft')
voting_s_clf=voting_s_clf.fit(x_train, y_train)
voting_soft_pred=voting_s_clf.predict(x_test)
```


```python
print('decision tree accuracy: ', accuracy_score(y_test, dt_pred))
print('KNN accuracy: ', accuracy_score(y_test, knn_pred))
print('SVM accuracy: ', accuracy_score(y_test, svm_pred))
print('voting hard accuracy: ', accuracy_score(y_test, voting_hard_pred))
print('voting soft accuracy: ', accuracy_score(y_test, voting_soft_pred))
```

    decision tree accuracy:  0.8481481481481481
    KNN accuracy:  0.9185185185185185
    SVM accuracy:  0.9851851851851852
    voting hard accuracy:  0.9592592592592593
    voting soft accuracy:  0.9518518518518518
    


```python
# visualization models accuracy
import matplotlib.pyplot as plt
import numpy as np

x=np.arange(5)
plt.bar(x, height=[accuracy_score(y_test, dt_pred), accuracy_score(y_test, knn_pred),
                   accuracy_score(y_test, svm_pred), accuracy_score(y_test, voting_hard_pred),
                   accuracy_score(y_test, voting_soft_pred)])
plt.xticks(x, ['decision tree', 'KNN', 'SVM', 'hard voting', 'soft voting'])
```




    ([<matplotlib.axis.XTick at 0x19d70304370>,
      <matplotlib.axis.XTick at 0x19d70304340>,
      <matplotlib.axis.XTick at 0x19d702c1b50>,
      <matplotlib.axis.XTick at 0x19d70357730>,
      <matplotlib.axis.XTick at 0x19d70357e80>],
     [Text(0, 0, 'decision tree'),
      Text(1, 0, 'KNN'),
      Text(2, 0, 'SVM'),
      Text(3, 0, 'hard voting'),
      Text(4, 0, 'soft voting')])




    
![png](output_10_1.png)
    

