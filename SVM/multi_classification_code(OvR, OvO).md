```python
# import libraries
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
```

MNIST datset을 불러온 후 각각의 방법으로 적용해보자<br>

MNIST datset을 훈련데이터:테스트데이터=7:3으로 분류한다.


```python
# data import, preprocessing
mnist=fetch_openml('mnist_784', version=1)
print(list(mnist))
x, y=mnist['data'], mnist['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
```

    ['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url']
    

---
### OvR


```python
# OVR
from sklearn.multiclass import OneVsRestClassifier

ovr_model=OneVsRestClassifier(SVC())
ovr_model.fit(x_train, y_train)

ovr_y_pred=ovr_model.predict(x_test)
ovr_y_true=y_test
```


```python
# OVR accuracy
ovr_acc=accuracy_score(ovr_y_true, ovr_y_pred)
ovr_acc
```




    0.9793333333333333




```python
# OVR confusion matrix
ovr_con_mat=confusion_matrix(ovr_y_true, ovr_y_pred)
ovr_con_mat
```




    array([[2075,    2,    0,    0,    0,    2,    8,    2,    7,    1],
           [   1, 2373,    8,    2,    6,    2,    2,    6,    1,    0],
           [   3,    2, 2045,    4,    5,    0,    3,   10,    8,    2],
           [   1,    0,   14, 2062,    0,   15,    0,   12,   16,    9],
           [   1,    4,    2,    0, 2041,    0,   10,    2,    5,   19],
           [   7,    1,    2,   12,    3, 1836,   10,    1,    9,    2],
           [   7,    1,    1,    1,    2,    7, 2050,    0,    3,    0],
           [   1,    6,   17,    2,   11,    1,    1, 2047,    4,   12],
           [   2,    8,    7,   11,    4,    2,    8,    1, 2034,    7],
           [   8,    1,    0,    7,   19,   10,    0,   12,    6, 2003]])




```python
plt.matshow(ovr_con_mat, cmap=plt.cm.gray)
```




    <matplotlib.image.AxesImage at 0x7fced1d0bd10>




    
![png](output_7_1.png)
    


---
### OvO


```python
# OVO
from sklearn.multiclass import OneVsOneClassifier

ovo_model=OneVsOneClassifier(SVC())
ovo_model.fit(x_train, y_train)

ovo_y_pred=ovo_model.predict(x_test)
ovo_y_true=y_test
```


```python
# OVO accuracy
ovo_acc=accuracy_score(ovo_y_true, ovo_y_pred)
ovo_acc
```




    0.9781904761904762




```python
# OVO confusion matrix
ovo_con_mat=confusion_matrix(ovo_y_true, ovo_y_pred)
ovo_con_mat
```




    array([[2072,    1,    1,    0,    2,    4,    4,    1,   11,    1],
           [   1, 2377,    3,    5,    4,    1,    1,    5,    2,    2],
           [   2,    2, 2041,    3,    8,    0,    4,   12,    8,    2],
           [   0,    0,   13, 2053,    0,   21,    1,   11,   23,    7],
           [   2,    2,    4,    0, 2042,    1,   10,    2,    3,   18],
           [   5,    1,    1,   15,    2, 1836,   12,    0,    8,    3],
           [   8,    1,    1,    0,    2,    7, 2049,    0,    4,    0],
           [   1,    4,   18,    1,   13,    1,    0, 2046,    4,   14],
           [   2,    9,    8,   13,    5,    7,    7,    3, 2025,    5],
           [   7,    1,    1,    8,   23,    6,    0,   11,    8, 2001]])




```python
plt.matshow(ovo_con_mat, cmap=plt.cm.gray)
```




    <matplotlib.image.AxesImage at 0x7fced1d33a90>




    
![png](output_12_1.png)
    

