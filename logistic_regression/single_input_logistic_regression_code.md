# 단일 입력 로지스틱 회귀(single input logistic regression)


```python
# import library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
```


```python
# making data
x=np.array([-5, -3.9, -2, -1.4, -0.89, 0.25, 0.38, 1.7, 3.5, 4.6])
y=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
```


```python
# logistic regression
model=Sequential()
model.add(Dense(input_dim=1, units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])
```


```python
# train
model.fit(x, y, epochs=200, verbose=0)

model.predict([-5, -3.9, -2, -1.4, -0.89, 0.25, 0.38, 1.7, 3.5, 4.6])
model.predict([-1000, 1000])
```




    array([[0.],
           [1.]], dtype=float32)




```python
# model summary
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 1)                 2         
                                                                     
     activation (Activation)     (None, 1)                 0         
                                                                     
    =================================================================
    Total params: 2
    Trainable params: 2
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.layers[0].get_weights()
```




    [array([[1.6471989]], dtype=float32), array([0.08966716], dtype=float32)]


