# 소프트맥스 다중 분류 로지스틱 회귀(multi input multi output logistic regression)-MNIST dataset


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

model=Sequential()
model.add(Dense(input_dim=input_dim, units=10, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
# train
model.fit(x_train, y_train, batch_size=1024, epochs=100)
```

    Epoch 1/100
    59/59 [==============================] - 2s 7ms/step - loss: 2.0527 - accuracy: 0.3099
    Epoch 2/100
    59/59 [==============================] - 0s 6ms/step - loss: 1.6394 - accuracy: 0.5990
    Epoch 3/100
    59/59 [==============================] - 0s 6ms/step - loss: 1.3771 - accuracy: 0.7017
    Epoch 4/100
    59/59 [==============================] - 0s 5ms/step - loss: 1.2010 - accuracy: 0.7501
    Epoch 5/100
    59/59 [==============================] - 0s 7ms/step - loss: 1.0769 - accuracy: 0.7772
    Epoch 6/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.9853 - accuracy: 0.7953
    Epoch 7/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.9151 - accuracy: 0.8075
    Epoch 8/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.8595 - accuracy: 0.8169
    Epoch 9/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.8144 - accuracy: 0.8248
    Epoch 10/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.7769 - accuracy: 0.8299
    Epoch 11/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.7454 - accuracy: 0.8354
    Epoch 12/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.7183 - accuracy: 0.8397
    Epoch 13/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.6949 - accuracy: 0.8427
    Epoch 14/100
    59/59 [==============================] - 0s 8ms/step - loss: 0.6743 - accuracy: 0.8459
    Epoch 15/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.6562 - accuracy: 0.8487
    Epoch 16/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.6400 - accuracy: 0.8506
    Epoch 17/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.6254 - accuracy: 0.8528
    Epoch 18/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.6123 - accuracy: 0.8547
    Epoch 19/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.6003 - accuracy: 0.8566
    Epoch 20/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.5894 - accuracy: 0.8582
    Epoch 21/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.5793 - accuracy: 0.8598
    Epoch 22/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.5701 - accuracy: 0.8615
    Epoch 23/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.5615 - accuracy: 0.8628
    Epoch 24/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.5536 - accuracy: 0.8642
    Epoch 25/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.5462 - accuracy: 0.8654
    Epoch 26/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.5393 - accuracy: 0.8665
    Epoch 27/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.5328 - accuracy: 0.8676
    Epoch 28/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.5267 - accuracy: 0.8684
    Epoch 29/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.5210 - accuracy: 0.8694
    Epoch 30/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.5155 - accuracy: 0.8703
    Epoch 31/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.5104 - accuracy: 0.8710
    Epoch 32/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.5056 - accuracy: 0.8723
    Epoch 33/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.5010 - accuracy: 0.8729
    Epoch 34/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4966 - accuracy: 0.8737
    Epoch 35/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4924 - accuracy: 0.8744
    Epoch 36/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4885 - accuracy: 0.8750
    Epoch 37/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4847 - accuracy: 0.8757
    Epoch 38/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4811 - accuracy: 0.8762
    Epoch 39/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4776 - accuracy: 0.8768
    Epoch 40/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4743 - accuracy: 0.8774
    Epoch 41/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4711 - accuracy: 0.8780
    Epoch 42/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4680 - accuracy: 0.8786
    Epoch 43/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4651 - accuracy: 0.8790
    Epoch 44/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.4622 - accuracy: 0.8796
    Epoch 45/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.4595 - accuracy: 0.8801
    Epoch 46/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4569 - accuracy: 0.8808
    Epoch 47/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4543 - accuracy: 0.8809
    Epoch 48/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4519 - accuracy: 0.8818
    Epoch 49/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4495 - accuracy: 0.8821
    Epoch 50/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4472 - accuracy: 0.8826
    Epoch 51/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4450 - accuracy: 0.8829
    Epoch 52/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4428 - accuracy: 0.8834
    Epoch 53/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4407 - accuracy: 0.8837
    Epoch 54/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4387 - accuracy: 0.8841
    Epoch 55/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4368 - accuracy: 0.8844
    Epoch 56/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4348 - accuracy: 0.8849
    Epoch 57/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4330 - accuracy: 0.8852
    Epoch 58/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4312 - accuracy: 0.8855
    Epoch 59/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4294 - accuracy: 0.8860
    Epoch 60/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4277 - accuracy: 0.8862
    Epoch 61/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4261 - accuracy: 0.8866
    Epoch 62/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4245 - accuracy: 0.8869
    Epoch 63/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4229 - accuracy: 0.8873
    Epoch 64/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4213 - accuracy: 0.8875
    Epoch 65/100
    59/59 [==============================] - 0s 4ms/step - loss: 0.4199 - accuracy: 0.8879
    Epoch 66/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4184 - accuracy: 0.8881
    Epoch 67/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4170 - accuracy: 0.8885
    Epoch 68/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4156 - accuracy: 0.8888
    Epoch 69/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4142 - accuracy: 0.8889
    Epoch 70/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4129 - accuracy: 0.8891
    Epoch 71/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.4116 - accuracy: 0.8893
    Epoch 72/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4103 - accuracy: 0.8896
    Epoch 73/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4091 - accuracy: 0.8899
    Epoch 74/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4078 - accuracy: 0.8901
    Epoch 75/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4067 - accuracy: 0.8903
    Epoch 76/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4055 - accuracy: 0.8906
    Epoch 77/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4043 - accuracy: 0.8910
    Epoch 78/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4032 - accuracy: 0.8912
    Epoch 79/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4021 - accuracy: 0.8914
    Epoch 80/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.4011 - accuracy: 0.8917
    Epoch 81/100
    59/59 [==============================] - 0s 7ms/step - loss: 0.4000 - accuracy: 0.8920
    Epoch 82/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.3990 - accuracy: 0.8921
    Epoch 83/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.3979 - accuracy: 0.8924
    Epoch 84/100
    59/59 [==============================] - 0s 6ms/step - loss: 0.3970 - accuracy: 0.8928
    Epoch 85/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3960 - accuracy: 0.8929
    Epoch 86/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3950 - accuracy: 0.8931
    Epoch 87/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3941 - accuracy: 0.8933
    Epoch 88/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3932 - accuracy: 0.8935
    Epoch 89/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3922 - accuracy: 0.8939
    Epoch 90/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3914 - accuracy: 0.8939
    Epoch 91/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3905 - accuracy: 0.8941
    Epoch 92/100
    59/59 [==============================] - 0s 4ms/step - loss: 0.3896 - accuracy: 0.8942
    Epoch 93/100
    59/59 [==============================] - 0s 4ms/step - loss: 0.3888 - accuracy: 0.8945
    Epoch 94/100
    59/59 [==============================] - 0s 4ms/step - loss: 0.3879 - accuracy: 0.8945
    Epoch 95/100
    59/59 [==============================] - 0s 4ms/step - loss: 0.3871 - accuracy: 0.8948
    Epoch 96/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3863 - accuracy: 0.8952
    Epoch 97/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3855 - accuracy: 0.8953
    Epoch 98/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3847 - accuracy: 0.8955
    Epoch 99/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3840 - accuracy: 0.8957
    Epoch 100/100
    59/59 [==============================] - 0s 5ms/step - loss: 0.3832 - accuracy: 0.8959
    




    <keras.callbacks.History at 0x2c353c14ca0>




```python
# test (accuracy)
score=model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])
```

    313/313 [==============================] - 1s 2ms/step - loss: 0.3645 - accuracy: 0.9027
    Test accuracy: 0.9027000069618225
    


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
# weight
model.layers[0].get_weights()
```




    [array([[ 0.06412486,  0.07372899,  0.08088329, ..., -0.05017535,
              0.04669949, -0.0121237 ],
            [ 0.05109195, -0.08491652,  0.01561028, ..., -0.02260319,
              0.0199421 , -0.05327139],
            [ 0.08498248,  0.01073638,  0.06480744, ..., -0.07710403,
              0.04558606,  0.05305208],
            ...,
            [ 0.07195922,  0.08411659, -0.00091064, ...,  0.08143271,
              0.05367707, -0.00808117],
            [ 0.0623358 , -0.05156458,  0.03020009, ...,  0.05795213,
             -0.07559095,  0.05025464],
            [ 0.01966112,  0.01868155,  0.07765353, ..., -0.01661055,
             -0.01637654, -0.04658562]], dtype=float32),
     array([-0.1087833 ,  0.21060227, -0.01805869, -0.07445134,  0.05033116,
             0.28069243, -0.02645303,  0.16966163, -0.41418687, -0.06935344],
           dtype=float32)]


