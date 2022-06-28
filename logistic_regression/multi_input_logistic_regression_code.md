# 다중 입력 로지스틱 회귀(multi input logistic regression)


```python
# import library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
```


```python
# making data
x=np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y=np.array([0, 0, 0, 1])
```


```python
# logistic regression
model=Sequential()
model.add(Dense(input_dim=2, units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])
```


```python
# train
model.fit(x, y, epochs=7000, verbose=0)

model.predict(x)
```




    array([[0.0164876 ],
           [0.17790183],
           [0.18112189],
           [0.74060816]], dtype=float32)




```python
# model summary
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 1)                 3         
                                                                     
     activation (Activation)     (None, 1)                 0         
                                                                     
    =================================================================
    Total params: 3
    Trainable params: 3
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.layers[0].get_weights()
```




    [array([[2.57976  ],
            [2.5578973]], dtype=float32),
     array([-4.0885253], dtype=float32)]


