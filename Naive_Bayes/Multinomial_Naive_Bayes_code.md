# Multinomial Naive Bayes


```python
# import library
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```


```python
# making data
review_list=[{'drama_review': 'this drama is so impressive', 'type': 'positive'},
             {'drama_review': 'i love that actor', 'type': 'positive'},
             {'drama_review': 'i want to recommend this drama', 'type': 'positive'},
             {'drama_review': 'my family like to see it on time', 'type': 'positive'},
             {'drama_review': 'this actor is so handsome', 'type': 'positive'},
             {'drama_review': 'actress is terrible', 'type': 'negative'},
             {'drama_review': 'actor is not acting well in this drama', 'type': 'negative'},
             {'drama_review': 'nobody will see this drama', 'type': 'negative'},
             {'drama_review': 'it was terrible', 'type': 'negative'},
             {'drama_review': 'do not recommend this drama', 'type': 'negative'}]
df=pd.DataFrame(review_list)
df
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
      <th>drama_review</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>this drama is so impressive</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>i love that actor</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>i want to recommend this drama</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>my family like to see it on time</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>this actor is so handsome</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>5</th>
      <td>actress is terrible</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>6</th>
      <td>actor is not acting well in this drama</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nobody will see this drama</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>8</th>
      <td>it was terrible</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>9</th>
      <td>do not recommend this drama</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
# data preprocessing
df['label']=df['type'].map({'positive': 1, 'negative': 0})
df_x=df['drama_review']
df_y=df['label']
```


```python
# CountVectorizer로 다항분포 나이브 베이즈의 입력 데이터를 고정된 크기의 벡터로서 각 인덱스는 단어의 빈도수를 나타나게 해줌
cv=CountVectorizer()
x_traincv=cv.fit_transform(df_x)
encoded_input=x_traincv.toarray()
encoded_input
```




    array([[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            1, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 0, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
            0, 1, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0]], dtype=int64)




```python
cv.get_feature_names()
```




    ['acting',
     'actor',
     'actress',
     'do',
     'drama',
     'family',
     'handsome',
     'impressive',
     'in',
     'is',
     'it',
     'like',
     'love',
     'my',
     'nobody',
     'not',
     'on',
     'recommend',
     'see',
     'so',
     'terrible',
     'that',
     'this',
     'time',
     'to',
     'want',
     'was',
     'well',
     'will']




```python
# train
mnb=MultinomialNB()
y_train=df_y.astype('int')
mnb.fit(x_traincv, y_train)
```




    MultinomialNB()




```python
# making test data
test_feedback_list=[{'drama_review': 'actress is acting so well', 'type': 'positive'},
                    {'drama_review': 'we all see this drama', 'type': 'positive'},
                    {'drama_review': 'my friend is big fan of this drama', 'type': 'positive'},
                    {'drama_review': 'drama has no story', 'type': 'negative'},
                    {'drama_review': 'my boyfriend hate to see that actor', 'type': 'negative'},
                    {'drama_review': 'i regret to see this drama', 'type': 'negative'}]
test_df=pd.DataFrame(test_feedback_list)
test_df['label']=test_df['type'].map({'positive':1, 'negative':0})

x_test=test_df['drama_review']
y_test=test_df['label']
```


```python
# test
x_testcv=cv.transform(x_test)
y_pred=mnb.predict(x_testcv)
```


```python
# accuracy
accuracy_score(y_test, y_pred)
```




    0.3333333333333333


