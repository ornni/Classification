# Bernoulli Naive Bayes


```python
# import library
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
```


```python
# making data
text_list=[{'text title': 'game sale only today', 'spam': True},
           {'text title': 'cheap ticket', 'spam': True},
           {'text title': 'limited time, right now', 'spam': True},
           {'text title': 'hot deal baby', 'spam': True},
           {'text title': 'meeting schedule', 'spam': False},
           {'text title': 'professor wants to see you', 'spam': False},
           {'text title': 'this month card bill', 'spam': False},
           {'text title': 'school class next semester', 'spam': False}]
df=pd.DataFrame(text_list)
```


```python
# data preprocessing
df['label']=df['spam'].map({True:1, False:0})

df_x=df['text title']
df_y=df['label']
```


```python
# CountVectorizer로 특정 데이터 안의 모든 단어를 포함한 고정 길이 벡터 만들기
# binary=True라고 하면 텍스트 제목에 특정 단어가 출현할 경우 무조건 1, 아니면 0을 갖도록 설정
cv=CountVectorizer(binary=True)
x_traincv=cv.fit_transform(df_x)
```


```python
# text_list vector encoding check
encoded_input=x_traincv.toarray()
encoded_input
```




    array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 1, 1],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0]], dtype=int64)




```python
# 고정된 벡터의 각 인덱스가 어떤 단어를 의미하는지
cv.get_feature_names()
```




    ['baby',
     'bill',
     'card',
     'cheap',
     'class',
     'deal',
     'game',
     'hot',
     'limited',
     'meeting',
     'month',
     'next',
     'now',
     'only',
     'professor',
     'right',
     'sale',
     'schedule',
     'school',
     'see',
     'semester',
     'this',
     'ticket',
     'time',
     'to',
     'today',
     'wants',
     'you']




```python
# Bernoulli Naive Bayes model
bnb=BernoulliNB()
y_train=df_y.astype('int')
bnb.fit(x_traincv, y_train)
```




    BernoulliNB()




```python
# making test data
test_text_list=[{'text title': 'game recommendation', 'spam': True},
                {'text title': 'cheap class for sale', 'spam': True},
                {'text title': 'health club hot deal until tommorrow', 'spam': True},
                {'text title': 'next month card bill', 'spam': False},
                {'text title': 'company meeting with professor', 'spam': False},
                {'text title': 'school team class', 'spam': False}]
test_df=pd.DataFrame(test_text_list)
test_df['label']=test_df['spam'].map({True:1, False:0})
x_test=test_df['text title']
y_test=test_df['label']
x_testcv=cv.transform(x_test)
```


```python
# test
y_pred=bnb.predict(x_testcv)
```


```python
# accuracy
accuracy_score(y_test, y_pred)
```




    1.0


