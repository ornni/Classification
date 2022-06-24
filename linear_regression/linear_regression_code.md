# linear regression


```python
# import library
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
```


```python
# make data
x=np.linspace(0, 20, 20)
y=x + np.random.randn(x.shape[0])
```

---
**linear regression model(y=ax)**


```python
model=Sequential()
model.add(Dense(input_dim=1, units=1, activation='linear', use_bias=False))

sgd=optimizers.SGD(lr=0.001)
model.compile(optimizer='sgd', loss='mse')
```

    C:\Users\user\anaconda3\lib\site-packages\keras\optimizer_v2\gradient_descent.py:102: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    


```python
# initial weight check
weights=model.layers[0].get_weights()
w=weights[0][0][0]
print('initial w is : ' + str(w))
```

    initial w is : 1.6560541
    


```python
# train
model.fit(x, y, batch_size=5, epochs=10)
```

    Epoch 1/10
    4/4 [==============================] - 0s 1ms/step - loss: 177.6688
    Epoch 2/10
    4/4 [==============================] - 0s 2ms/step - loss: 46241.1016
    Epoch 3/10
    4/4 [==============================] - 0s 2ms/step - loss: 2072210.0000
    Epoch 4/10
    4/4 [==============================] - 0s 1ms/step - loss: 8820661.0000
    Epoch 5/10
    4/4 [==============================] - 0s 2ms/step - loss: 1688723072.0000
    Epoch 6/10
    4/4 [==============================] - 0s 2ms/step - loss: 27812964352.0000
    Epoch 7/10
    4/4 [==============================] - 0s 2ms/step - loss: 3183149318144.0000
    Epoch 8/10
    4/4 [==============================] - 0s 2ms/step - loss: 39708453240832.0000
    Epoch 9/10
    4/4 [==============================] - 0s 2ms/step - loss: 2219509981118464.0000
    Epoch 10/10
    4/4 [==============================] - 0s 2ms/step - loss: 3373743250341888.0000
    




    <keras.callbacks.History at 0x129d693e940>




```python
# visualize
import matplotlib.pyplot as plt

plt.plot(x, y, label='data')
plt.plot(x, w*x, label='prediction')
plt.legend()
plt.show()
```


    
![png](output_7_0.png)
    


---
**linear regression model(y=ax+b)**


```python
model=Sequential()
model.add(Dense(input_dim=1, units=1, activation='linear', use_bias=True))
sgd=optimizers.SGD(lr=0.001)
model.compile(optimizer='sgd', loss='mse')
```

    C:\Users\user\anaconda3\lib\site-packages\keras\optimizer_v2\gradient_descent.py:102: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    


```python
# initial weight check
weights=model.layers[0].get_weights()
w=weights[0][0][0]
print('initial w is : ' + str(w))
```

    initial w is : 1.2183813
    


```python
# train
model.fit(x, y, batch_size=5, epochs=10)
```

    Epoch 1/10
    4/4 [==============================] - 0s 1ms/step - loss: 1.3261
    Epoch 2/10
    4/4 [==============================] - 0s 2ms/step - loss: 7.5996
    Epoch 3/10
    4/4 [==============================] - 0s 1ms/step - loss: 207.2372
    Epoch 4/10
    4/4 [==============================] - 0s 2ms/step - loss: 8398.2930
    Epoch 5/10
    4/4 [==============================] - 0s 2ms/step - loss: 1973.5781
    Epoch 6/10
    4/4 [==============================] - 0s 1ms/step - loss: 13023.6816
    Epoch 7/10
    4/4 [==============================] - 0s 2ms/step - loss: 703998.8750
    Epoch 8/10
    4/4 [==============================] - 0s 1ms/step - loss: 28686144.0000
    Epoch 9/10
    4/4 [==============================] - 0s 2ms/step - loss: 1125682432.0000
    Epoch 10/10
    4/4 [==============================] - 0s 2ms/step - loss: 119611219968.0000
    




    <keras.callbacks.History at 0x129df095d60>




```python
# visualize
import matplotlib.pyplot as plt

plt.plot(x, y, label='data')
plt.plot(x, w*x, label='prediction')
plt.legend()
plt.show()
```


    
![png](output_12_0.png)
    



```python

```
