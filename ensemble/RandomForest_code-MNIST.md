# RandomForest-MNIST dataset


```python
# import library
from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```


```python
# load data
mnist=datasets.load_digits()
data, target=mnist.data, mnist.target
```


```python
# cross validation
def cross_validation(classifier, data, target):
    cv_scores=[]
    for i in range(10):
        scores=cross_val_score(classifier, data, target, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    return cv_scores
```


```python
# decision making tree's MNIST validation score
dt_cv_scores=cross_validation(tree.DecisionTreeClassifier(), data, target)
```


```python
# randomforest tree's MNIST validation score
rf_cv_scores=cross_validation(RandomForestClassifier(), data, target)
```


```python
# decision making tree VS randomforest tree accuracy visualization
cv_list=[['random_forest', rf_cv_scores],
         ['decision_tree', dt_cv_scores]]

df=pd.DataFrame.from_dict(dict(cv_list))
df.plot()
```




    <AxesSubplot:>




    
![png](output_6_1.png)
    



```python
# accuracy
np.mean(dt_cv_scores)
np.mean(rf_cv_scores)
```




    0.9489621353196771


