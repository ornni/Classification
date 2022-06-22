# SVM-iris dataset


```python
# import library
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
```


```python
# data load
iris=load_iris()
df=pd.DataFrame(data=np.c_[iris.data, iris.target], columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# train, test 분리 (7:3)
x=df[df.columns[:-1]]
y=df['target']
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
```

    (105, 4) (105,) (45, 4) (45,)
    


```python
# 변수 별 관계 그래프
import seaborn as sns
sns.pairplot(data=df, hue='target', palette='Set1')
```




    <seaborn.axisgrid.PairGrid at 0x246b6b92880>




    
![png](output_4_1.png)
    


---
**최적의 parameter를 알아서 구해주는 SVM model**


```python
from sklearn.svm import SVC
model=SVC()
```


```python
# train
model.fit(x_train, y_train)
```




    SVC()




```python
# accuracy
from sklearn.metrics import accuracy_score
y_pred=model.predict(x_test)
accuracy_score(y_test, y_pred)
```




    0.9777777777777777




```python
# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```




    array([[22,  0,  0],
           [ 0, 12,  1],
           [ 0,  0, 10]], dtype=int64)



---
**GridSearch를 이용하는 SVM model**


```python
from sklearn.model_selection import GridSearchCV
def svc_param_selection(x, y, nfolds):
    svm_parameters=[{'kernel': ['rbf'],
                     'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                     'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    clf=GridSearchCV(SVC(), svm_parameters, cv=10)
    clf.fit(x_train, y_train.values.ravel())
    print(clf.best_params_)
    return clf

clf=svc_param_selection(x_train, y_train.values.ravel(), 10)
```

    {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
    

C=1, gamma=0.1에서 최적의 값을 가짐


```python
# accuracy
y_pred=clf.predict(x_test)
accuracy_score(y_test, y_pred)
```




    0.9777777777777777




```python
# confusion matrix
confusion_matrix(y_test, y_pred)
```




    array([[22,  0,  0],
           [ 0, 12,  1],
           [ 0,  0, 10]], dtype=int64)


