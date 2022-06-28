# 소프트맥스 다중 분류 로지스틱 회귀(multi input multi output logictic regression)-MNIST dataset


```python
# import library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
```


```python
# data
(x_train, y_train), (x_test, y_test)=mnist.load_data()
```


```python
# data normalization(0~1)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train /=255
x_test /=255

input_dim=28*28
x_train=x_train.reshape(60000, input_dim)
x_test=x_test.reshape(10000, input_dim)
```


```python
# softmax(one hot encoding)
num_classes=10
y_train=to_categorical(y_train, num_classes)
y_test=to_categorical(y_test, num_classes)

print(y_train[0])
```

    [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    


```python
model=Sequential()
model.add(Dense(input_dim=input_dim, units=10, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
# train
model.fit(x_train, y_train, batch_size=1024, epochs=100, verbose=0)
```




    <keras.callbacks.History at 0x17ed0fb5e50>




```python
# test
score=model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])
```

    313/313 [==============================] - 0s 1ms/step - loss: 0.3626 - accuracy: 0.9028
    Test accuracy: 0.9028000235557556
    


```python
# model summary
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 10)                7850      
                                                                     
    =================================================================
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.layers[0].get_weights()
```




    [array([[-0.04409529, -0.05895729,  0.03967334, ...,  0.04950198,
             -0.06400667,  0.0401521 ],
            [ 0.00041428,  0.05739426,  0.07228253, ..., -0.01974975,
             -0.01635892,  0.07450568],
            [ 0.05466763,  0.07676894,  0.0797663 , ..., -0.00863095,
              0.08436284,  0.05294173],
            ...,
            [-0.06429401, -0.01079037,  0.01253838, ...,  0.00322585,
              0.01265511,  0.01863956],
            [-0.06722592, -0.00338085, -0.0565906 , ...,  0.06552061,
             -0.08054182,  0.0243744 ],
            [-0.05497177,  0.01857083, -0.08356979, ..., -0.0043671 ,
             -0.0728386 , -0.06827633]], dtype=float32),
     array([-0.10706354,  0.21053435, -0.0399349 , -0.08550763,  0.06749034,
             0.27650246, -0.02149145,  0.15165763, -0.41032502, -0.04186268],
           dtype=float32)]


