# K mean cluster


```python
# import library
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```


```python
# making data
df=pd.DataFrame({'height': [150, 170, 156, 164, 173, 166, 154, 181, 183, 179, 175], 
                 'weight': [45, 62, 48, 60, 63, 53, 40, 78, 83, 69, 80]})
```


```python
# data visualization
plt.scatter(x=df['height'], y=df['weight'])
```




    <matplotlib.collections.PathCollection at 0x204116e39d0>




    
![data](https://github.com/ornni/ML_algorithm/blob/main/cluster/image/kmean_cluster_code_output_3_1.png?raw=true)
    



```python
# kmean
data_points=df.values
model=KMeans(n_clusters=4)
model.fit(df)

pred=model.predict(df)
```


```python
# kmean visualization
plt.scatter(x=df['height'], y=df['weight'], c=pred)
```




    <matplotlib.collections.PathCollection at 0x204122c1d30>




    
![result](https://github.com/ornni/ML_algorithm/blob/main/cluster/image/kmean_cluster_code_output_5_1.png?raw=true)
    

