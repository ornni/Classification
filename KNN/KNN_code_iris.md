# KNN-iris dataset


```python
# load library
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
# target 개수 확인
df.target.value_counts()
```




    0.0    50
    1.0    50
    2.0    50
    Name: target, dtype: int64




```python
# train, test 분리 (7:3)
x=df[df.columns[:-1]]
y=df['target']
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
```

    (105, 4) (105,) (45, 4) (45,)
    


```python
# scale문제 해결을 위해 정규화
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

pd.DataFrame(x_train).head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.194444</td>
      <td>0.000000</td>
      <td>0.446429</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.111111</td>
      <td>0.500000</td>
      <td>0.107143</td>
      <td>0.041667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.388889</td>
      <td>0.333333</td>
      <td>0.553571</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.555556</td>
      <td>0.333333</td>
      <td>0.732143</td>
      <td>0.583333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.555556</td>
      <td>0.583333</td>
      <td>0.821429</td>
      <td>0.958333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# test 데이터 정규화 진행
x_test=scaler.transform(x_test)
```

---
**K=3일 때 KNN model**


```python
# KNN model
from sklearn.neighbors import KNeighborsClassifier

k=3
model=KNeighborsClassifier(n_neighbors=k)
```


```python
# train
model.fit(x_train, y_train)
```




    KNeighborsClassifier(n_neighbors=3)




```python
# accuracy
from sklearn.metrics import accuracy_score

y_pred=model.predict(x_test)
accuracy_score(y_test, y_pred)
```




    0.9777777777777777



---
**K가 3~55사이의 값일 때 KNN model**


```python
import matplotlib.pyplot as plt

k_range=range(3, 55)

accuracy_list=[]

for k in k_range:
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
```


```python
# 시각화
plt.plot(k_range, accuracy_list)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
```


    
![png](output_13_0.png)
    



```python
# 최적의 k값
k=k_range[accuracy_list.index(max(accuracy_list))]
print('answer of k:' + str(k))
```

    answer of k:6
    

---
**k가 3~(학습데이터 개수/2) 사이의 값일 때 KNN model (교차검증)**


```python
from sklearn.model_selection import cross_val_score

max_k_range=x_train.shape[0]//2
k_list=[]

for i in range(3, max_k_range, 2):
    k_list.append(i)

cross_validation_scores=[]
y_train.values.ravel()

for k in k_list:
    model=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(model, x_train, y_train.values.ravel(), cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())
```


```python
# 시각화
plt.plot(k_list, cross_validation_scores)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
```


    
![png](output_17_0.png)
    



```python
# 최적의 k값
k=k_list[cross_validation_scores.index(max(cross_validation_scores))]
print('answer of k:' + str(k))
```

    answer of k:13
    
