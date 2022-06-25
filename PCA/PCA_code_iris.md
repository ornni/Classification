# PCA-IRIS dataset
**2차원으로 사영하기**


```python
# import library
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
```


```python
# load data
iris=load_iris()
df=pd.DataFrame(data=np.c_[iris.data, iris.target], columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
```


```python
# data preprocessing
x=df[df.columns[:-1]]
y=df['target']
```


```python
# data normalization
from sklearn.preprocessing import StandardScaler

x_std=StandardScaler().fit_transform(x)
```


```python
# covariance matrix
cov_matrix=np.cov(x_std.T)
```


```python
# eigen vector, eigen value
eig_vals, eig_vecs=np.linalg.eig(cov_matrix)
```


```python
# 가장 큰 고유벡터로 데이터를 사영할 경우 유지되는 정보의 정도
eig_vals[0]/sum(eig_vals)
```




    0.7296244541329988




```python
# 데이터 고유벡터로 사영시키기
projected_x0=x_std.dot(eig_vecs.T[0])/np.linalg.norm(eig_vecs.T[0])
projected_x1=x_std.dot(eig_vecs.T[1])/np.linalg.norm(eig_vecs.T[0])
```


```python
# visualization
result=pd.DataFrame(projected_x0, columns=['PC0'])
result['PC1']=projected_x1
result['target']=df['target']

sns.scatterplot(x='PC0', y='PC1', data=result, hue=result['target'], s=100)
```




    <AxesSubplot:xlabel='PC0', ylabel='PC1'>




    
![png](https://github.com/ornni/ML_algorithm/blob/main/PCA/image/PCA_code_iris_output_9_1.png?raw=true)
    


---
**1차원으로 사영하기**


```python
result['y-axis']=0.0
sns.lmplot('PC0', 'y-axis', data=result, hue='target')
```

    C:\Users\user\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <seaborn.axisgrid.FacetGrid at 0x20433061c70>




    
![png](https://github.com/ornni/ML_algorithm/blob/main/PCA/image/PCA_code_iris_output_11_2.png?raw=true)
    


---
**sklearn으로 간단하게 PCA구현**


```python
from sklearn import decomposition
pca=decomposition.PCA(n_components=1)
sklearn_pca_x=pca.fit_transform(x_std)

sklearn_result=pd.DataFrame(sklearn_pca_x, columns=['PC1'])
sklearn_result['y-axis']=0.0
sklearn_result['target']=df['target']
sns.lmplot('PC1', 'y-axis', data=sklearn_result, hue='target')
```

    C:\Users\user\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <seaborn.axisgrid.FacetGrid at 0x2043346fc40>




    
![png](https://github.com/ornni/ML_algorithm/blob/main/PCA/image/PCA_code_iris_output_13_2.png?raw=true)
    

