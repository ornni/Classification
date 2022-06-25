# Decision Tree-IRIS dataset


```python
# import library
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
```


```python
# load data
iris=load_iris()
x_train, x_test, y_train, y_test=train_test_split(iris.data, iris.target, test_size=0.3)
```


```python
# Decision Tree Classifier model
model=DecisionTreeClassifier()

model.fit(x_train, y_train)

y_pred=model.predict(x_test)
```


```python
# accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```




    0.9111111111111111


