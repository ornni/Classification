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
model.fit(x, y, epochs=7000)

model.predict(x)
```

    Epoch 1/7000
    1/1 [==============================] - 0s 323ms/step - loss: 0.7212 - binary_accuracy: 0.7500
    Epoch 2/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7209 - binary_accuracy: 0.7500
    Epoch 3/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7206 - binary_accuracy: 0.7500
    Epoch 4/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7202 - binary_accuracy: 0.7500
    Epoch 5/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7199 - binary_accuracy: 0.7500
    Epoch 6/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7195 - binary_accuracy: 0.7500
    Epoch 7/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7192 - binary_accuracy: 0.7500
    Epoch 8/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7189 - binary_accuracy: 0.7500
    Epoch 9/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7185 - binary_accuracy: 0.7500
    Epoch 10/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7182 - binary_accuracy: 0.7500
    Epoch 11/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7178 - binary_accuracy: 0.7500
    Epoch 12/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7175 - binary_accuracy: 0.7500
    Epoch 13/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7172 - binary_accuracy: 0.7500
    Epoch 14/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7168 - binary_accuracy: 0.7500
    Epoch 15/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7165 - binary_accuracy: 0.7500
    Epoch 16/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7162 - binary_accuracy: 0.7500
    Epoch 17/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7158 - binary_accuracy: 0.7500
    Epoch 18/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7155 - binary_accuracy: 0.7500
    Epoch 19/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7152 - binary_accuracy: 0.7500
    Epoch 20/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7148 - binary_accuracy: 0.7500
    Epoch 21/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7145 - binary_accuracy: 0.7500
    Epoch 22/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7142 - binary_accuracy: 0.7500
    Epoch 23/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7138 - binary_accuracy: 0.7500
    Epoch 24/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7135 - binary_accuracy: 0.7500
    Epoch 25/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7132 - binary_accuracy: 0.7500
    Epoch 26/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7129 - binary_accuracy: 0.7500
    Epoch 27/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7125 - binary_accuracy: 0.7500
    Epoch 28/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7122 - binary_accuracy: 0.7500
    Epoch 29/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7119 - binary_accuracy: 0.7500
    Epoch 30/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7115 - binary_accuracy: 0.7500
    Epoch 31/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7112 - binary_accuracy: 0.7500
    Epoch 32/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7109 - binary_accuracy: 0.7500
    Epoch 33/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7106 - binary_accuracy: 0.7500
    Epoch 34/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7102 - binary_accuracy: 0.7500
    Epoch 35/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7099 - binary_accuracy: 0.7500
    Epoch 36/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7096 - binary_accuracy: 0.7500
    Epoch 37/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7092 - binary_accuracy: 0.7500
    Epoch 38/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7089 - binary_accuracy: 0.7500
    Epoch 39/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7086 - binary_accuracy: 0.7500
    Epoch 40/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7083 - binary_accuracy: 0.7500
    Epoch 41/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7080 - binary_accuracy: 0.7500
    Epoch 42/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.7076 - binary_accuracy: 0.7500
    Epoch 43/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7073 - binary_accuracy: 0.7500
    Epoch 44/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7070 - binary_accuracy: 0.7500
    Epoch 45/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7067 - binary_accuracy: 0.7500
    Epoch 46/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7063 - binary_accuracy: 0.7500
    Epoch 47/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7060 - binary_accuracy: 0.7500
    Epoch 48/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7057 - binary_accuracy: 0.7500
    Epoch 49/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.7054 - binary_accuracy: 0.7500
    Epoch 50/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7051 - binary_accuracy: 0.7500
    Epoch 51/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.7047 - binary_accuracy: 0.7500
    Epoch 52/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7044 - binary_accuracy: 0.7500
    Epoch 53/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7041 - binary_accuracy: 0.7500
    Epoch 54/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7038 - binary_accuracy: 0.7500
    Epoch 55/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7035 - binary_accuracy: 0.7500
    Epoch 56/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7031 - binary_accuracy: 0.7500
    Epoch 57/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7028 - binary_accuracy: 0.7500
    Epoch 58/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7025 - binary_accuracy: 0.7500
    Epoch 59/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7022 - binary_accuracy: 0.7500
    Epoch 60/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7019 - binary_accuracy: 0.7500
    Epoch 61/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7016 - binary_accuracy: 0.7500
    Epoch 62/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7012 - binary_accuracy: 0.7500
    Epoch 63/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7009 - binary_accuracy: 0.7500
    Epoch 64/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7006 - binary_accuracy: 0.7500
    Epoch 65/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.7003 - binary_accuracy: 0.7500
    Epoch 66/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.7000 - binary_accuracy: 0.7500
    Epoch 67/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6997 - binary_accuracy: 0.7500
    Epoch 68/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6994 - binary_accuracy: 0.7500
    Epoch 69/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6990 - binary_accuracy: 0.7500
    Epoch 70/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6987 - binary_accuracy: 0.7500
    Epoch 71/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6984 - binary_accuracy: 0.7500
    Epoch 72/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6981 - binary_accuracy: 0.7500
    Epoch 73/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6978 - binary_accuracy: 0.7500
    Epoch 74/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6975 - binary_accuracy: 0.7500
    Epoch 75/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6972 - binary_accuracy: 0.7500
    Epoch 76/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6969 - binary_accuracy: 0.7500
    Epoch 77/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6966 - binary_accuracy: 0.7500
    Epoch 78/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6962 - binary_accuracy: 0.7500
    Epoch 79/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6959 - binary_accuracy: 0.7500
    Epoch 80/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6956 - binary_accuracy: 0.7500
    Epoch 81/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6953 - binary_accuracy: 0.7500
    Epoch 82/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6950 - binary_accuracy: 0.7500
    Epoch 83/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6947 - binary_accuracy: 0.7500
    Epoch 84/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6944 - binary_accuracy: 0.7500
    Epoch 85/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6941 - binary_accuracy: 0.7500
    Epoch 86/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6938 - binary_accuracy: 0.7500
    Epoch 87/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6935 - binary_accuracy: 0.7500
    Epoch 88/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6932 - binary_accuracy: 0.7500
    Epoch 89/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6929 - binary_accuracy: 0.7500
    Epoch 90/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6926 - binary_accuracy: 0.7500
    Epoch 91/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6923 - binary_accuracy: 0.7500
    Epoch 92/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6920 - binary_accuracy: 0.7500
    Epoch 93/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6916 - binary_accuracy: 0.7500
    Epoch 94/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6913 - binary_accuracy: 0.7500
    Epoch 95/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6910 - binary_accuracy: 0.7500
    Epoch 96/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6907 - binary_accuracy: 0.7500
    Epoch 97/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6904 - binary_accuracy: 0.7500
    Epoch 98/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6901 - binary_accuracy: 0.7500
    Epoch 99/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6898 - binary_accuracy: 0.7500
    Epoch 100/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6895 - binary_accuracy: 0.7500
    Epoch 101/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6892 - binary_accuracy: 0.7500
    Epoch 102/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6889 - binary_accuracy: 0.7500
    Epoch 103/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6886 - binary_accuracy: 0.7500
    Epoch 104/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6883 - binary_accuracy: 0.7500
    Epoch 105/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.6880 - binary_accuracy: 0.7500
    Epoch 106/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.6877 - binary_accuracy: 0.7500
    Epoch 107/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6874 - binary_accuracy: 0.7500
    Epoch 108/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.6871 - binary_accuracy: 0.7500
    Epoch 109/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6868 - binary_accuracy: 0.7500
    Epoch 110/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6865 - binary_accuracy: 0.7500
    Epoch 111/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6862 - binary_accuracy: 0.7500
    Epoch 112/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6859 - binary_accuracy: 0.7500
    Epoch 113/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6856 - binary_accuracy: 0.7500
    Epoch 114/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6853 - binary_accuracy: 0.7500
    Epoch 115/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6850 - binary_accuracy: 0.7500
    Epoch 116/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6847 - binary_accuracy: 0.7500
    Epoch 117/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6844 - binary_accuracy: 0.7500
    Epoch 118/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6841 - binary_accuracy: 0.7500
    Epoch 119/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6839 - binary_accuracy: 0.7500
    Epoch 120/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6836 - binary_accuracy: 0.7500
    Epoch 121/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6833 - binary_accuracy: 0.7500
    Epoch 122/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6830 - binary_accuracy: 0.7500
    Epoch 123/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6827 - binary_accuracy: 0.7500
    Epoch 124/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6824 - binary_accuracy: 0.7500
    Epoch 125/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6821 - binary_accuracy: 0.7500
    Epoch 126/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6818 - binary_accuracy: 0.7500
    Epoch 127/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6815 - binary_accuracy: 0.7500
    Epoch 128/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6812 - binary_accuracy: 0.7500
    Epoch 129/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6809 - binary_accuracy: 0.7500
    Epoch 130/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6806 - binary_accuracy: 0.7500
    Epoch 131/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6803 - binary_accuracy: 0.7500
    Epoch 132/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6800 - binary_accuracy: 0.7500
    Epoch 133/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6797 - binary_accuracy: 0.7500
    Epoch 134/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6794 - binary_accuracy: 0.7500
    Epoch 135/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6792 - binary_accuracy: 0.7500
    Epoch 136/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6789 - binary_accuracy: 0.7500
    Epoch 137/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6786 - binary_accuracy: 0.7500
    Epoch 138/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6783 - binary_accuracy: 0.7500
    Epoch 139/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6780 - binary_accuracy: 0.7500
    Epoch 140/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6777 - binary_accuracy: 0.7500
    Epoch 141/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6774 - binary_accuracy: 0.7500
    Epoch 142/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6771 - binary_accuracy: 0.7500
    Epoch 143/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6768 - binary_accuracy: 0.7500
    Epoch 144/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6766 - binary_accuracy: 0.7500
    Epoch 145/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6763 - binary_accuracy: 0.7500
    Epoch 146/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6760 - binary_accuracy: 0.7500
    Epoch 147/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6757 - binary_accuracy: 0.7500
    Epoch 148/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6754 - binary_accuracy: 0.7500
    Epoch 149/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6751 - binary_accuracy: 0.7500
    Epoch 150/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6748 - binary_accuracy: 0.7500
    Epoch 151/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6745 - binary_accuracy: 0.7500
    Epoch 152/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6743 - binary_accuracy: 0.7500
    Epoch 153/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6740 - binary_accuracy: 0.7500
    Epoch 154/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6737 - binary_accuracy: 0.7500
    Epoch 155/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6734 - binary_accuracy: 0.7500
    Epoch 156/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6731 - binary_accuracy: 0.7500
    Epoch 157/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6728 - binary_accuracy: 0.7500
    Epoch 158/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6725 - binary_accuracy: 0.7500
    Epoch 159/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6723 - binary_accuracy: 0.7500
    Epoch 160/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6720 - binary_accuracy: 0.7500
    Epoch 161/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6717 - binary_accuracy: 0.7500
    Epoch 162/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6714 - binary_accuracy: 0.7500
    Epoch 163/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6711 - binary_accuracy: 0.7500
    Epoch 164/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6708 - binary_accuracy: 0.7500
    Epoch 165/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6706 - binary_accuracy: 0.7500
    Epoch 166/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6703 - binary_accuracy: 0.7500
    Epoch 167/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6700 - binary_accuracy: 0.7500
    Epoch 168/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6697 - binary_accuracy: 0.7500
    Epoch 169/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6694 - binary_accuracy: 0.7500
    Epoch 170/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6692 - binary_accuracy: 0.7500
    Epoch 171/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6689 - binary_accuracy: 0.7500
    Epoch 172/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6686 - binary_accuracy: 0.7500
    Epoch 173/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6683 - binary_accuracy: 0.7500
    Epoch 174/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6680 - binary_accuracy: 0.7500
    Epoch 175/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6678 - binary_accuracy: 0.7500
    Epoch 176/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6675 - binary_accuracy: 0.7500
    Epoch 177/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6672 - binary_accuracy: 0.7500
    Epoch 178/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6669 - binary_accuracy: 0.7500
    Epoch 179/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6666 - binary_accuracy: 0.7500
    Epoch 180/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6664 - binary_accuracy: 0.7500
    Epoch 181/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6661 - binary_accuracy: 0.7500
    Epoch 182/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6658 - binary_accuracy: 0.7500
    Epoch 183/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6655 - binary_accuracy: 0.7500
    Epoch 184/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6652 - binary_accuracy: 0.7500
    Epoch 185/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6650 - binary_accuracy: 0.7500
    Epoch 186/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6647 - binary_accuracy: 0.7500
    Epoch 187/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6644 - binary_accuracy: 0.7500
    Epoch 188/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6641 - binary_accuracy: 0.7500
    Epoch 189/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6639 - binary_accuracy: 0.7500
    Epoch 190/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6636 - binary_accuracy: 0.7500
    Epoch 191/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6633 - binary_accuracy: 0.7500
    Epoch 192/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6630 - binary_accuracy: 0.7500
    Epoch 193/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6628 - binary_accuracy: 0.7500
    Epoch 194/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6625 - binary_accuracy: 0.7500
    Epoch 195/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6622 - binary_accuracy: 0.7500
    Epoch 196/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6619 - binary_accuracy: 0.7500
    Epoch 197/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6617 - binary_accuracy: 0.7500
    Epoch 198/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6614 - binary_accuracy: 0.7500
    Epoch 199/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6611 - binary_accuracy: 0.7500
    Epoch 200/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6608 - binary_accuracy: 0.7500
    Epoch 201/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6606 - binary_accuracy: 0.7500
    Epoch 202/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6603 - binary_accuracy: 0.7500
    Epoch 203/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6600 - binary_accuracy: 0.7500
    Epoch 204/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6598 - binary_accuracy: 0.7500
    Epoch 205/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6595 - binary_accuracy: 0.7500
    Epoch 206/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6592 - binary_accuracy: 0.7500
    Epoch 207/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6589 - binary_accuracy: 0.7500
    Epoch 208/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6587 - binary_accuracy: 0.7500
    Epoch 209/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6584 - binary_accuracy: 0.7500
    Epoch 210/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6581 - binary_accuracy: 0.7500
    Epoch 211/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6579 - binary_accuracy: 0.7500
    Epoch 212/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6576 - binary_accuracy: 0.7500
    Epoch 213/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6573 - binary_accuracy: 0.7500
    Epoch 214/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6570 - binary_accuracy: 0.7500
    Epoch 215/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6568 - binary_accuracy: 0.7500
    Epoch 216/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6565 - binary_accuracy: 0.7500
    Epoch 217/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6562 - binary_accuracy: 0.7500
    Epoch 218/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6560 - binary_accuracy: 0.7500
    Epoch 219/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6557 - binary_accuracy: 0.7500
    Epoch 220/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6554 - binary_accuracy: 0.7500
    Epoch 221/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6552 - binary_accuracy: 0.7500
    Epoch 222/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6549 - binary_accuracy: 0.7500
    Epoch 223/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6546 - binary_accuracy: 0.7500
    Epoch 224/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6544 - binary_accuracy: 0.7500
    Epoch 225/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6541 - binary_accuracy: 0.7500
    Epoch 226/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6538 - binary_accuracy: 0.7500
    Epoch 227/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6536 - binary_accuracy: 0.7500
    Epoch 228/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6533 - binary_accuracy: 0.7500
    Epoch 229/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6530 - binary_accuracy: 0.7500
    Epoch 230/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6528 - binary_accuracy: 0.7500
    Epoch 231/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6525 - binary_accuracy: 0.7500
    Epoch 232/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6522 - binary_accuracy: 0.7500
    Epoch 233/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6520 - binary_accuracy: 0.7500
    Epoch 234/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6517 - binary_accuracy: 0.7500
    Epoch 235/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6514 - binary_accuracy: 0.7500
    Epoch 236/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6512 - binary_accuracy: 0.7500
    Epoch 237/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6509 - binary_accuracy: 0.7500
    Epoch 238/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6506 - binary_accuracy: 0.7500
    Epoch 239/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6504 - binary_accuracy: 0.7500
    Epoch 240/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6501 - binary_accuracy: 0.7500
    Epoch 241/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6499 - binary_accuracy: 0.7500
    Epoch 242/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6496 - binary_accuracy: 0.7500
    Epoch 243/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6493 - binary_accuracy: 0.7500
    Epoch 244/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6491 - binary_accuracy: 0.7500
    Epoch 245/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6488 - binary_accuracy: 0.7500
    Epoch 246/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6485 - binary_accuracy: 0.7500
    Epoch 247/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6483 - binary_accuracy: 0.7500
    Epoch 248/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6480 - binary_accuracy: 0.7500
    Epoch 249/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6478 - binary_accuracy: 0.7500
    Epoch 250/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6475 - binary_accuracy: 0.7500
    Epoch 251/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6472 - binary_accuracy: 0.7500
    Epoch 252/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6470 - binary_accuracy: 0.7500
    Epoch 253/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6467 - binary_accuracy: 0.7500
    Epoch 254/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6465 - binary_accuracy: 0.7500
    Epoch 255/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6462 - binary_accuracy: 0.7500
    Epoch 256/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6459 - binary_accuracy: 0.7500
    Epoch 257/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6457 - binary_accuracy: 0.7500
    Epoch 258/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6454 - binary_accuracy: 0.7500
    Epoch 259/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6452 - binary_accuracy: 0.7500
    Epoch 260/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6449 - binary_accuracy: 0.7500
    Epoch 261/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6446 - binary_accuracy: 0.7500
    Epoch 262/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6444 - binary_accuracy: 0.7500
    Epoch 263/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6441 - binary_accuracy: 0.7500
    Epoch 264/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6439 - binary_accuracy: 0.7500
    Epoch 265/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6436 - binary_accuracy: 0.7500
    Epoch 266/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6433 - binary_accuracy: 0.7500
    Epoch 267/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6431 - binary_accuracy: 0.7500
    Epoch 268/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6428 - binary_accuracy: 0.7500
    Epoch 269/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6426 - binary_accuracy: 0.7500
    Epoch 270/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6423 - binary_accuracy: 0.7500
    Epoch 271/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6421 - binary_accuracy: 0.7500
    Epoch 272/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6418 - binary_accuracy: 0.7500
    Epoch 273/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6416 - binary_accuracy: 0.7500
    Epoch 274/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6413 - binary_accuracy: 0.7500
    Epoch 275/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6410 - binary_accuracy: 0.7500
    Epoch 276/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6408 - binary_accuracy: 0.7500
    Epoch 277/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6405 - binary_accuracy: 0.7500
    Epoch 278/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6403 - binary_accuracy: 0.7500
    Epoch 279/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6400 - binary_accuracy: 0.7500
    Epoch 280/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6398 - binary_accuracy: 0.7500
    Epoch 281/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6395 - binary_accuracy: 0.7500
    Epoch 282/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6393 - binary_accuracy: 0.7500
    Epoch 283/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6390 - binary_accuracy: 0.7500
    Epoch 284/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6387 - binary_accuracy: 0.7500
    Epoch 285/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6385 - binary_accuracy: 0.7500
    Epoch 286/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6382 - binary_accuracy: 0.7500
    Epoch 287/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6380 - binary_accuracy: 0.7500
    Epoch 288/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6377 - binary_accuracy: 0.7500
    Epoch 289/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6375 - binary_accuracy: 0.7500
    Epoch 290/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6372 - binary_accuracy: 0.7500
    Epoch 291/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6370 - binary_accuracy: 0.7500
    Epoch 292/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6367 - binary_accuracy: 0.7500
    Epoch 293/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6365 - binary_accuracy: 0.7500
    Epoch 294/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6362 - binary_accuracy: 0.7500
    Epoch 295/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6360 - binary_accuracy: 0.7500
    Epoch 296/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6357 - binary_accuracy: 0.7500
    Epoch 297/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6355 - binary_accuracy: 0.7500
    Epoch 298/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6352 - binary_accuracy: 0.7500
    Epoch 299/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6350 - binary_accuracy: 0.7500
    Epoch 300/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6347 - binary_accuracy: 0.7500
    Epoch 301/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6345 - binary_accuracy: 0.7500
    Epoch 302/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6342 - binary_accuracy: 0.7500
    Epoch 303/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6340 - binary_accuracy: 0.7500
    Epoch 304/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6337 - binary_accuracy: 0.7500
    Epoch 305/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6335 - binary_accuracy: 0.7500
    Epoch 306/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6332 - binary_accuracy: 0.7500
    Epoch 307/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6330 - binary_accuracy: 0.7500
    Epoch 308/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6327 - binary_accuracy: 0.7500
    Epoch 309/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6325 - binary_accuracy: 0.7500
    Epoch 310/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6322 - binary_accuracy: 0.7500
    Epoch 311/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6320 - binary_accuracy: 0.7500
    Epoch 312/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6317 - binary_accuracy: 0.7500
    Epoch 313/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6315 - binary_accuracy: 0.7500
    Epoch 314/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6312 - binary_accuracy: 0.7500
    Epoch 315/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6310 - binary_accuracy: 0.7500
    Epoch 316/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6307 - binary_accuracy: 0.7500
    Epoch 317/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6305 - binary_accuracy: 0.7500
    Epoch 318/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6302 - binary_accuracy: 0.7500
    Epoch 319/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6300 - binary_accuracy: 0.7500
    Epoch 320/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6297 - binary_accuracy: 0.7500
    Epoch 321/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6295 - binary_accuracy: 0.7500
    Epoch 322/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6293 - binary_accuracy: 0.7500
    Epoch 323/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6290 - binary_accuracy: 0.7500
    Epoch 324/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6288 - binary_accuracy: 0.7500
    Epoch 325/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6285 - binary_accuracy: 0.7500
    Epoch 326/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6283 - binary_accuracy: 0.7500
    Epoch 327/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6280 - binary_accuracy: 0.7500
    Epoch 328/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6278 - binary_accuracy: 0.7500
    Epoch 329/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6275 - binary_accuracy: 0.7500
    Epoch 330/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6273 - binary_accuracy: 0.7500
    Epoch 331/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6271 - binary_accuracy: 0.7500
    Epoch 332/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6268 - binary_accuracy: 0.7500
    Epoch 333/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6266 - binary_accuracy: 0.7500
    Epoch 334/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6263 - binary_accuracy: 0.7500
    Epoch 335/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6261 - binary_accuracy: 0.7500
    Epoch 336/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6258 - binary_accuracy: 0.7500
    Epoch 337/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6256 - binary_accuracy: 0.7500
    Epoch 338/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6253 - binary_accuracy: 0.7500
    Epoch 339/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6251 - binary_accuracy: 0.7500
    Epoch 340/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6249 - binary_accuracy: 0.7500
    Epoch 341/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6246 - binary_accuracy: 0.7500
    Epoch 342/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6244 - binary_accuracy: 0.7500
    Epoch 343/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6241 - binary_accuracy: 0.7500
    Epoch 344/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6239 - binary_accuracy: 0.7500
    Epoch 345/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6237 - binary_accuracy: 0.7500
    Epoch 346/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6234 - binary_accuracy: 0.7500
    Epoch 347/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6232 - binary_accuracy: 0.7500
    Epoch 348/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6229 - binary_accuracy: 0.7500
    Epoch 349/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6227 - binary_accuracy: 0.7500
    Epoch 350/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6224 - binary_accuracy: 0.7500
    Epoch 351/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6222 - binary_accuracy: 0.7500
    Epoch 352/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6220 - binary_accuracy: 0.7500
    Epoch 353/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6217 - binary_accuracy: 0.7500
    Epoch 354/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6215 - binary_accuracy: 0.7500
    Epoch 355/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6213 - binary_accuracy: 0.7500
    Epoch 356/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6210 - binary_accuracy: 0.7500
    Epoch 357/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6208 - binary_accuracy: 0.7500
    Epoch 358/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6205 - binary_accuracy: 0.7500
    Epoch 359/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6203 - binary_accuracy: 0.7500
    Epoch 360/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6201 - binary_accuracy: 0.7500
    Epoch 361/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6198 - binary_accuracy: 0.7500
    Epoch 362/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6196 - binary_accuracy: 0.7500
    Epoch 363/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6193 - binary_accuracy: 0.7500
    Epoch 364/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6191 - binary_accuracy: 0.7500
    Epoch 365/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6189 - binary_accuracy: 0.7500
    Epoch 366/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6186 - binary_accuracy: 0.7500
    Epoch 367/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6184 - binary_accuracy: 0.7500
    Epoch 368/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6182 - binary_accuracy: 0.7500
    Epoch 369/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6179 - binary_accuracy: 0.7500
    Epoch 370/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6177 - binary_accuracy: 0.7500
    Epoch 371/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6174 - binary_accuracy: 0.7500
    Epoch 372/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6172 - binary_accuracy: 0.7500
    Epoch 373/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6170 - binary_accuracy: 0.7500
    Epoch 374/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6167 - binary_accuracy: 0.7500
    Epoch 375/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6165 - binary_accuracy: 0.7500
    Epoch 376/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.6163 - binary_accuracy: 0.7500
    Epoch 377/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6160 - binary_accuracy: 0.7500
    Epoch 378/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6158 - binary_accuracy: 0.7500
    Epoch 379/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6156 - binary_accuracy: 0.7500
    Epoch 380/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6153 - binary_accuracy: 0.7500
    Epoch 381/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6151 - binary_accuracy: 0.7500
    Epoch 382/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6149 - binary_accuracy: 0.7500
    Epoch 383/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6146 - binary_accuracy: 0.7500
    Epoch 384/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6144 - binary_accuracy: 0.7500
    Epoch 385/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6141 - binary_accuracy: 0.7500
    Epoch 386/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6139 - binary_accuracy: 0.7500
    Epoch 387/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6137 - binary_accuracy: 0.7500
    Epoch 388/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6134 - binary_accuracy: 0.7500
    Epoch 389/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6132 - binary_accuracy: 0.7500
    Epoch 390/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6130 - binary_accuracy: 0.7500
    Epoch 391/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6127 - binary_accuracy: 0.7500
    Epoch 392/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6125 - binary_accuracy: 0.7500
    Epoch 393/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6123 - binary_accuracy: 0.7500
    Epoch 394/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6121 - binary_accuracy: 0.7500
    Epoch 395/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6118 - binary_accuracy: 0.7500
    Epoch 396/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6116 - binary_accuracy: 0.7500
    Epoch 397/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6114 - binary_accuracy: 0.7500
    Epoch 398/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6111 - binary_accuracy: 0.7500
    Epoch 399/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6109 - binary_accuracy: 0.7500
    Epoch 400/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6107 - binary_accuracy: 0.7500
    Epoch 401/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6104 - binary_accuracy: 0.7500
    Epoch 402/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6102 - binary_accuracy: 0.7500
    Epoch 403/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6100 - binary_accuracy: 0.7500
    Epoch 404/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6097 - binary_accuracy: 0.7500
    Epoch 405/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.6095 - binary_accuracy: 0.7500
    Epoch 406/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.6093 - binary_accuracy: 0.7500
    Epoch 407/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.6090 - binary_accuracy: 0.7500
    Epoch 408/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6088 - binary_accuracy: 0.7500
    Epoch 409/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6086 - binary_accuracy: 0.7500
    Epoch 410/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6084 - binary_accuracy: 0.7500
    Epoch 411/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.6081 - binary_accuracy: 0.7500
    Epoch 412/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6079 - binary_accuracy: 0.7500
    Epoch 413/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.6077 - binary_accuracy: 0.7500
    Epoch 414/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6074 - binary_accuracy: 0.7500
    Epoch 415/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6072 - binary_accuracy: 0.7500
    Epoch 416/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6070 - binary_accuracy: 0.7500
    Epoch 417/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6068 - binary_accuracy: 0.7500
    Epoch 418/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6065 - binary_accuracy: 0.7500
    Epoch 419/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6063 - binary_accuracy: 0.7500
    Epoch 420/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6061 - binary_accuracy: 0.7500
    Epoch 421/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6058 - binary_accuracy: 0.7500
    Epoch 422/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6056 - binary_accuracy: 0.7500
    Epoch 423/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6054 - binary_accuracy: 0.7500
    Epoch 424/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6052 - binary_accuracy: 0.7500
    Epoch 425/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6049 - binary_accuracy: 0.7500
    Epoch 426/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6047 - binary_accuracy: 0.7500
    Epoch 427/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6045 - binary_accuracy: 0.7500
    Epoch 428/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6043 - binary_accuracy: 0.7500
    Epoch 429/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6040 - binary_accuracy: 0.7500
    Epoch 430/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6038 - binary_accuracy: 0.7500
    Epoch 431/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6036 - binary_accuracy: 0.7500
    Epoch 432/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6034 - binary_accuracy: 0.7500
    Epoch 433/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6031 - binary_accuracy: 0.7500
    Epoch 434/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6029 - binary_accuracy: 0.7500
    Epoch 435/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6027 - binary_accuracy: 0.7500
    Epoch 436/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6025 - binary_accuracy: 0.7500
    Epoch 437/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6022 - binary_accuracy: 0.7500
    Epoch 438/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6020 - binary_accuracy: 0.7500
    Epoch 439/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6018 - binary_accuracy: 0.7500
    Epoch 440/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6016 - binary_accuracy: 0.7500
    Epoch 441/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6013 - binary_accuracy: 0.7500
    Epoch 442/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6011 - binary_accuracy: 0.7500
    Epoch 443/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6009 - binary_accuracy: 0.7500
    Epoch 444/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6007 - binary_accuracy: 0.7500
    Epoch 445/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6004 - binary_accuracy: 0.7500
    Epoch 446/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.6002 - binary_accuracy: 0.7500
    Epoch 447/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.6000 - binary_accuracy: 0.7500
    Epoch 448/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5998 - binary_accuracy: 0.7500
    Epoch 449/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5995 - binary_accuracy: 0.7500
    Epoch 450/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5993 - binary_accuracy: 0.7500
    Epoch 451/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5991 - binary_accuracy: 0.7500
    Epoch 452/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5989 - binary_accuracy: 0.7500
    Epoch 453/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.5986 - binary_accuracy: 0.7500
    Epoch 454/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5984 - binary_accuracy: 0.7500
    Epoch 455/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5982 - binary_accuracy: 0.7500
    Epoch 456/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5980 - binary_accuracy: 0.7500
    Epoch 457/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5978 - binary_accuracy: 0.7500
    Epoch 458/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5975 - binary_accuracy: 0.7500
    Epoch 459/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5973 - binary_accuracy: 0.7500
    Epoch 460/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5971 - binary_accuracy: 0.7500
    Epoch 461/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5969 - binary_accuracy: 0.7500
    Epoch 462/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5967 - binary_accuracy: 0.7500
    Epoch 463/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5964 - binary_accuracy: 0.7500
    Epoch 464/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5962 - binary_accuracy: 0.7500
    Epoch 465/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5960 - binary_accuracy: 0.7500
    Epoch 466/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5958 - binary_accuracy: 0.7500
    Epoch 467/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5956 - binary_accuracy: 0.7500
    Epoch 468/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5953 - binary_accuracy: 0.7500
    Epoch 469/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5951 - binary_accuracy: 0.7500
    Epoch 470/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5949 - binary_accuracy: 0.7500
    Epoch 471/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5947 - binary_accuracy: 0.7500
    Epoch 472/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5945 - binary_accuracy: 0.7500
    Epoch 473/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5942 - binary_accuracy: 0.7500
    Epoch 474/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5940 - binary_accuracy: 0.7500
    Epoch 475/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5938 - binary_accuracy: 0.7500
    Epoch 476/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5936 - binary_accuracy: 0.7500
    Epoch 477/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5934 - binary_accuracy: 0.7500
    Epoch 478/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5931 - binary_accuracy: 0.7500
    Epoch 479/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5929 - binary_accuracy: 0.7500
    Epoch 480/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5927 - binary_accuracy: 0.7500
    Epoch 481/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5925 - binary_accuracy: 0.7500
    Epoch 482/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5923 - binary_accuracy: 0.7500
    Epoch 483/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5921 - binary_accuracy: 0.7500
    Epoch 484/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5918 - binary_accuracy: 0.7500
    Epoch 485/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5916 - binary_accuracy: 0.7500
    Epoch 486/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5914 - binary_accuracy: 0.7500
    Epoch 487/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5912 - binary_accuracy: 0.7500
    Epoch 488/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5910 - binary_accuracy: 0.7500
    Epoch 489/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5908 - binary_accuracy: 0.7500
    Epoch 490/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5905 - binary_accuracy: 0.7500
    Epoch 491/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5903 - binary_accuracy: 0.7500
    Epoch 492/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5901 - binary_accuracy: 0.7500
    Epoch 493/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5899 - binary_accuracy: 0.7500
    Epoch 494/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5897 - binary_accuracy: 0.7500
    Epoch 495/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5895 - binary_accuracy: 0.7500
    Epoch 496/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5892 - binary_accuracy: 0.7500
    Epoch 497/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5890 - binary_accuracy: 0.7500
    Epoch 498/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5888 - binary_accuracy: 0.7500
    Epoch 499/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5886 - binary_accuracy: 0.7500
    Epoch 500/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5884 - binary_accuracy: 0.7500
    Epoch 501/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5882 - binary_accuracy: 0.7500
    Epoch 502/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5880 - binary_accuracy: 0.7500
    Epoch 503/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5877 - binary_accuracy: 0.7500
    Epoch 504/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5875 - binary_accuracy: 0.7500
    Epoch 505/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5873 - binary_accuracy: 0.7500
    Epoch 506/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5871 - binary_accuracy: 0.7500
    Epoch 507/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5869 - binary_accuracy: 0.7500
    Epoch 508/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5867 - binary_accuracy: 0.7500
    Epoch 509/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5865 - binary_accuracy: 0.7500
    Epoch 510/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5863 - binary_accuracy: 0.7500
    Epoch 511/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5860 - binary_accuracy: 0.7500
    Epoch 512/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5858 - binary_accuracy: 0.7500
    Epoch 513/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5856 - binary_accuracy: 0.7500
    Epoch 514/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5854 - binary_accuracy: 0.7500
    Epoch 515/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5852 - binary_accuracy: 0.7500
    Epoch 516/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5850 - binary_accuracy: 0.7500
    Epoch 517/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5848 - binary_accuracy: 0.7500
    Epoch 518/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5846 - binary_accuracy: 0.7500
    Epoch 519/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5843 - binary_accuracy: 0.7500
    Epoch 520/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5841 - binary_accuracy: 0.7500
    Epoch 521/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5839 - binary_accuracy: 0.7500
    Epoch 522/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5837 - binary_accuracy: 0.7500
    Epoch 523/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5835 - binary_accuracy: 0.7500
    Epoch 524/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5833 - binary_accuracy: 0.7500
    Epoch 525/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5831 - binary_accuracy: 0.7500
    Epoch 526/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5829 - binary_accuracy: 0.7500
    Epoch 527/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5827 - binary_accuracy: 0.7500
    Epoch 528/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5824 - binary_accuracy: 0.7500
    Epoch 529/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5822 - binary_accuracy: 0.7500
    Epoch 530/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5820 - binary_accuracy: 0.7500
    Epoch 531/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5818 - binary_accuracy: 0.7500
    Epoch 532/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5816 - binary_accuracy: 0.7500
    Epoch 533/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5814 - binary_accuracy: 0.7500
    Epoch 534/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5812 - binary_accuracy: 0.7500
    Epoch 535/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5810 - binary_accuracy: 0.7500
    Epoch 536/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5808 - binary_accuracy: 0.7500
    Epoch 537/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5806 - binary_accuracy: 0.7500
    Epoch 538/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5804 - binary_accuracy: 0.7500
    Epoch 539/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5801 - binary_accuracy: 0.7500
    Epoch 540/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5799 - binary_accuracy: 0.7500
    Epoch 541/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5797 - binary_accuracy: 0.7500
    Epoch 542/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5795 - binary_accuracy: 0.7500
    Epoch 543/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5793 - binary_accuracy: 0.7500
    Epoch 544/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5791 - binary_accuracy: 0.7500
    Epoch 545/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5789 - binary_accuracy: 0.7500
    Epoch 546/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5787 - binary_accuracy: 0.7500
    Epoch 547/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5785 - binary_accuracy: 0.7500
    Epoch 548/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5783 - binary_accuracy: 0.7500
    Epoch 549/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5781 - binary_accuracy: 0.7500
    Epoch 550/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5779 - binary_accuracy: 0.7500
    Epoch 551/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5777 - binary_accuracy: 0.7500
    Epoch 552/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5774 - binary_accuracy: 0.7500
    Epoch 553/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5772 - binary_accuracy: 0.7500
    Epoch 554/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5770 - binary_accuracy: 0.7500
    Epoch 555/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5768 - binary_accuracy: 0.7500
    Epoch 556/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5766 - binary_accuracy: 0.7500
    Epoch 557/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5764 - binary_accuracy: 0.7500
    Epoch 558/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5762 - binary_accuracy: 0.7500
    Epoch 559/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5760 - binary_accuracy: 0.7500
    Epoch 560/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5758 - binary_accuracy: 0.7500
    Epoch 561/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5756 - binary_accuracy: 0.7500
    Epoch 562/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5754 - binary_accuracy: 0.7500
    Epoch 563/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5752 - binary_accuracy: 0.7500
    Epoch 564/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5750 - binary_accuracy: 0.7500
    Epoch 565/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5748 - binary_accuracy: 0.7500
    Epoch 566/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5746 - binary_accuracy: 0.7500
    Epoch 567/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5744 - binary_accuracy: 0.7500
    Epoch 568/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5742 - binary_accuracy: 0.7500
    Epoch 569/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5740 - binary_accuracy: 0.7500
    Epoch 570/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.5738 - binary_accuracy: 0.7500
    Epoch 571/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5735 - binary_accuracy: 0.7500
    Epoch 572/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5733 - binary_accuracy: 0.7500
    Epoch 573/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5731 - binary_accuracy: 0.7500
    Epoch 574/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5729 - binary_accuracy: 0.7500
    Epoch 575/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5727 - binary_accuracy: 0.7500
    Epoch 576/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5725 - binary_accuracy: 0.7500
    Epoch 577/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5723 - binary_accuracy: 0.7500
    Epoch 578/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5721 - binary_accuracy: 0.7500
    Epoch 579/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5719 - binary_accuracy: 0.7500
    Epoch 580/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5717 - binary_accuracy: 0.7500
    Epoch 581/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5715 - binary_accuracy: 0.7500
    Epoch 582/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5713 - binary_accuracy: 0.7500
    Epoch 583/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5711 - binary_accuracy: 0.7500
    Epoch 584/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5709 - binary_accuracy: 0.7500
    Epoch 585/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5707 - binary_accuracy: 0.7500
    Epoch 586/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5705 - binary_accuracy: 0.7500
    Epoch 587/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5703 - binary_accuracy: 0.7500
    Epoch 588/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5701 - binary_accuracy: 0.7500
    Epoch 589/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5699 - binary_accuracy: 0.7500
    Epoch 590/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.5697 - binary_accuracy: 0.7500
    Epoch 591/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5695 - binary_accuracy: 0.7500
    Epoch 592/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5693 - binary_accuracy: 0.7500
    Epoch 593/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5691 - binary_accuracy: 0.7500
    Epoch 594/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5689 - binary_accuracy: 0.7500
    Epoch 595/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5687 - binary_accuracy: 0.7500
    Epoch 596/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5685 - binary_accuracy: 0.7500
    Epoch 597/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5683 - binary_accuracy: 0.7500
    Epoch 598/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5681 - binary_accuracy: 0.7500
    Epoch 599/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5679 - binary_accuracy: 0.7500
    Epoch 600/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5677 - binary_accuracy: 0.7500
    Epoch 601/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5675 - binary_accuracy: 0.7500
    Epoch 602/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5673 - binary_accuracy: 0.7500
    Epoch 603/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5671 - binary_accuracy: 0.7500
    Epoch 604/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5669 - binary_accuracy: 0.7500
    Epoch 605/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5667 - binary_accuracy: 0.7500
    Epoch 606/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5665 - binary_accuracy: 0.7500
    Epoch 607/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5663 - binary_accuracy: 0.7500
    Epoch 608/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5661 - binary_accuracy: 0.7500
    Epoch 609/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5659 - binary_accuracy: 0.7500
    Epoch 610/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5657 - binary_accuracy: 0.7500
    Epoch 611/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.5655 - binary_accuracy: 0.7500
    Epoch 612/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5653 - binary_accuracy: 0.7500
    Epoch 613/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5651 - binary_accuracy: 0.7500
    Epoch 614/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5649 - binary_accuracy: 0.7500
    Epoch 615/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5647 - binary_accuracy: 0.7500
    Epoch 616/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5645 - binary_accuracy: 0.7500
    Epoch 617/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5643 - binary_accuracy: 0.7500
    Epoch 618/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5641 - binary_accuracy: 0.7500
    Epoch 619/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5639 - binary_accuracy: 0.7500
    Epoch 620/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5637 - binary_accuracy: 0.7500
    Epoch 621/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5635 - binary_accuracy: 0.7500
    Epoch 622/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5633 - binary_accuracy: 0.7500
    Epoch 623/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5631 - binary_accuracy: 0.7500
    Epoch 624/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5630 - binary_accuracy: 0.7500
    Epoch 625/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5628 - binary_accuracy: 0.7500
    Epoch 626/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5626 - binary_accuracy: 0.7500
    Epoch 627/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5624 - binary_accuracy: 0.7500
    Epoch 628/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5622 - binary_accuracy: 0.7500
    Epoch 629/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5620 - binary_accuracy: 0.7500
    Epoch 630/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5618 - binary_accuracy: 0.7500
    Epoch 631/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5616 - binary_accuracy: 0.7500
    Epoch 632/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.5614 - binary_accuracy: 0.7500
    Epoch 633/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5612 - binary_accuracy: 0.7500
    Epoch 634/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5610 - binary_accuracy: 0.7500
    Epoch 635/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5608 - binary_accuracy: 0.7500
    Epoch 636/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5606 - binary_accuracy: 0.7500
    Epoch 637/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5604 - binary_accuracy: 0.7500
    Epoch 638/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5602 - binary_accuracy: 0.7500
    Epoch 639/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5600 - binary_accuracy: 0.7500
    Epoch 640/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5598 - binary_accuracy: 0.7500
    Epoch 641/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5596 - binary_accuracy: 0.7500
    Epoch 642/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5594 - binary_accuracy: 0.7500
    Epoch 643/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5592 - binary_accuracy: 0.7500
    Epoch 644/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5591 - binary_accuracy: 0.7500
    Epoch 645/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5589 - binary_accuracy: 0.7500
    Epoch 646/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5587 - binary_accuracy: 0.7500
    Epoch 647/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5585 - binary_accuracy: 0.7500
    Epoch 648/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5583 - binary_accuracy: 0.7500
    Epoch 649/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5581 - binary_accuracy: 0.7500
    Epoch 650/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5579 - binary_accuracy: 0.7500
    Epoch 651/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5577 - binary_accuracy: 0.7500
    Epoch 652/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5575 - binary_accuracy: 0.7500
    Epoch 653/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5573 - binary_accuracy: 0.7500
    Epoch 654/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5571 - binary_accuracy: 0.7500
    Epoch 655/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5569 - binary_accuracy: 0.7500
    Epoch 656/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5567 - binary_accuracy: 0.7500
    Epoch 657/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5566 - binary_accuracy: 0.7500
    Epoch 658/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5564 - binary_accuracy: 0.7500
    Epoch 659/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5562 - binary_accuracy: 0.7500
    Epoch 660/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5560 - binary_accuracy: 0.7500
    Epoch 661/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5558 - binary_accuracy: 0.7500
    Epoch 662/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5556 - binary_accuracy: 0.7500
    Epoch 663/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5554 - binary_accuracy: 0.7500
    Epoch 664/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5552 - binary_accuracy: 0.7500
    Epoch 665/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5550 - binary_accuracy: 0.7500
    Epoch 666/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5548 - binary_accuracy: 0.7500
    Epoch 667/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5546 - binary_accuracy: 0.7500
    Epoch 668/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5545 - binary_accuracy: 0.7500
    Epoch 669/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5543 - binary_accuracy: 0.7500
    Epoch 670/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5541 - binary_accuracy: 0.7500
    Epoch 671/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5539 - binary_accuracy: 0.7500
    Epoch 672/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5537 - binary_accuracy: 0.7500
    Epoch 673/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5535 - binary_accuracy: 0.7500
    Epoch 674/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5533 - binary_accuracy: 0.7500
    Epoch 675/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5531 - binary_accuracy: 0.7500
    Epoch 676/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5529 - binary_accuracy: 0.7500
    Epoch 677/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5527 - binary_accuracy: 0.7500
    Epoch 678/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5526 - binary_accuracy: 0.7500
    Epoch 679/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5524 - binary_accuracy: 0.7500
    Epoch 680/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5522 - binary_accuracy: 0.7500
    Epoch 681/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5520 - binary_accuracy: 0.7500
    Epoch 682/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5518 - binary_accuracy: 0.7500
    Epoch 683/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5516 - binary_accuracy: 0.7500
    Epoch 684/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5514 - binary_accuracy: 0.7500
    Epoch 685/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5512 - binary_accuracy: 0.7500
    Epoch 686/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5511 - binary_accuracy: 0.7500
    Epoch 687/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5509 - binary_accuracy: 0.7500
    Epoch 688/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5507 - binary_accuracy: 0.7500
    Epoch 689/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5505 - binary_accuracy: 0.7500
    Epoch 690/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5503 - binary_accuracy: 0.7500
    Epoch 691/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5501 - binary_accuracy: 0.7500
    Epoch 692/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5499 - binary_accuracy: 0.7500
    Epoch 693/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5497 - binary_accuracy: 0.7500
    Epoch 694/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5496 - binary_accuracy: 0.7500
    Epoch 695/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5494 - binary_accuracy: 0.7500
    Epoch 696/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5492 - binary_accuracy: 0.7500
    Epoch 697/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5490 - binary_accuracy: 0.7500
    Epoch 698/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5488 - binary_accuracy: 0.7500
    Epoch 699/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5486 - binary_accuracy: 0.7500
    Epoch 700/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5484 - binary_accuracy: 0.7500
    Epoch 701/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5482 - binary_accuracy: 0.7500
    Epoch 702/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5481 - binary_accuracy: 0.7500
    Epoch 703/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5479 - binary_accuracy: 0.7500
    Epoch 704/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5477 - binary_accuracy: 0.7500
    Epoch 705/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5475 - binary_accuracy: 0.7500
    Epoch 706/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5473 - binary_accuracy: 0.7500
    Epoch 707/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5471 - binary_accuracy: 0.7500
    Epoch 708/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5469 - binary_accuracy: 0.7500
    Epoch 709/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5468 - binary_accuracy: 0.7500
    Epoch 710/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5466 - binary_accuracy: 0.7500
    Epoch 711/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5464 - binary_accuracy: 0.7500
    Epoch 712/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5462 - binary_accuracy: 0.7500
    Epoch 713/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5460 - binary_accuracy: 0.7500
    Epoch 714/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5458 - binary_accuracy: 0.7500
    Epoch 715/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5457 - binary_accuracy: 0.7500
    Epoch 716/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5455 - binary_accuracy: 0.7500
    Epoch 717/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5453 - binary_accuracy: 0.7500
    Epoch 718/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.5451 - binary_accuracy: 0.7500
    Epoch 719/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5449 - binary_accuracy: 0.7500
    Epoch 720/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5447 - binary_accuracy: 0.7500
    Epoch 721/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5446 - binary_accuracy: 0.7500
    Epoch 722/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5444 - binary_accuracy: 0.7500
    Epoch 723/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5442 - binary_accuracy: 0.7500
    Epoch 724/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5440 - binary_accuracy: 0.7500
    Epoch 725/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5438 - binary_accuracy: 0.7500
    Epoch 726/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5436 - binary_accuracy: 0.7500
    Epoch 727/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5435 - binary_accuracy: 0.7500
    Epoch 728/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5433 - binary_accuracy: 0.7500
    Epoch 729/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5431 - binary_accuracy: 0.7500
    Epoch 730/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5429 - binary_accuracy: 0.7500
    Epoch 731/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5427 - binary_accuracy: 0.7500
    Epoch 732/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5425 - binary_accuracy: 0.7500
    Epoch 733/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5424 - binary_accuracy: 0.7500
    Epoch 734/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5422 - binary_accuracy: 0.7500
    Epoch 735/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5420 - binary_accuracy: 0.7500
    Epoch 736/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5418 - binary_accuracy: 0.7500
    Epoch 737/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5416 - binary_accuracy: 0.7500
    Epoch 738/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5415 - binary_accuracy: 0.7500
    Epoch 739/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.5413 - binary_accuracy: 0.7500
    Epoch 740/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5411 - binary_accuracy: 0.7500
    Epoch 741/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5409 - binary_accuracy: 0.7500
    Epoch 742/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5407 - binary_accuracy: 0.7500
    Epoch 743/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5406 - binary_accuracy: 0.7500
    Epoch 744/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5404 - binary_accuracy: 0.7500
    Epoch 745/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5402 - binary_accuracy: 0.7500
    Epoch 746/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5400 - binary_accuracy: 0.7500
    Epoch 747/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5398 - binary_accuracy: 0.7500
    Epoch 748/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5397 - binary_accuracy: 0.7500
    Epoch 749/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5395 - binary_accuracy: 0.7500
    Epoch 750/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5393 - binary_accuracy: 0.7500
    Epoch 751/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5391 - binary_accuracy: 0.7500
    Epoch 752/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5389 - binary_accuracy: 0.7500
    Epoch 753/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5388 - binary_accuracy: 0.7500
    Epoch 754/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5386 - binary_accuracy: 0.7500
    Epoch 755/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5384 - binary_accuracy: 0.7500
    Epoch 756/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5382 - binary_accuracy: 0.7500
    Epoch 757/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5380 - binary_accuracy: 0.7500
    Epoch 758/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5379 - binary_accuracy: 0.7500
    Epoch 759/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5377 - binary_accuracy: 0.7500
    Epoch 760/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.5375 - binary_accuracy: 0.7500
    Epoch 761/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5373 - binary_accuracy: 0.7500
    Epoch 762/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5371 - binary_accuracy: 0.7500
    Epoch 763/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5370 - binary_accuracy: 0.7500
    Epoch 764/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5368 - binary_accuracy: 0.7500
    Epoch 765/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5366 - binary_accuracy: 0.7500
    Epoch 766/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5364 - binary_accuracy: 0.7500
    Epoch 767/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5363 - binary_accuracy: 0.7500
    Epoch 768/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5361 - binary_accuracy: 0.7500
    Epoch 769/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5359 - binary_accuracy: 0.7500
    Epoch 770/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5357 - binary_accuracy: 0.7500
    Epoch 771/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.5355 - binary_accuracy: 0.7500
    Epoch 772/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5354 - binary_accuracy: 0.7500
    Epoch 773/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5352 - binary_accuracy: 0.7500
    Epoch 774/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5350 - binary_accuracy: 0.7500
    Epoch 775/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5348 - binary_accuracy: 0.7500
    Epoch 776/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5347 - binary_accuracy: 0.7500
    Epoch 777/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5345 - binary_accuracy: 0.7500
    Epoch 778/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5343 - binary_accuracy: 0.7500
    Epoch 779/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5341 - binary_accuracy: 0.7500
    Epoch 780/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5340 - binary_accuracy: 0.7500
    Epoch 781/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5338 - binary_accuracy: 0.7500
    Epoch 782/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.5336 - binary_accuracy: 0.7500
    Epoch 783/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5334 - binary_accuracy: 0.7500
    Epoch 784/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5333 - binary_accuracy: 0.7500
    Epoch 785/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5331 - binary_accuracy: 0.7500
    Epoch 786/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5329 - binary_accuracy: 0.7500
    Epoch 787/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5327 - binary_accuracy: 0.7500
    Epoch 788/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5326 - binary_accuracy: 0.7500
    Epoch 789/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5324 - binary_accuracy: 0.7500
    Epoch 790/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5322 - binary_accuracy: 0.7500
    Epoch 791/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5320 - binary_accuracy: 0.7500
    Epoch 792/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5319 - binary_accuracy: 0.7500
    Epoch 793/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.5317 - binary_accuracy: 0.7500
    Epoch 794/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5315 - binary_accuracy: 0.7500
    Epoch 795/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5313 - binary_accuracy: 0.7500
    Epoch 796/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5312 - binary_accuracy: 0.7500
    Epoch 797/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5310 - binary_accuracy: 0.7500
    Epoch 798/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5308 - binary_accuracy: 0.7500
    Epoch 799/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5306 - binary_accuracy: 0.7500
    Epoch 800/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5305 - binary_accuracy: 0.7500
    Epoch 801/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5303 - binary_accuracy: 0.7500
    Epoch 802/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5301 - binary_accuracy: 0.7500
    Epoch 803/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.5299 - binary_accuracy: 0.7500
    Epoch 804/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5298 - binary_accuracy: 0.7500
    Epoch 805/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5296 - binary_accuracy: 0.7500
    Epoch 806/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5294 - binary_accuracy: 0.7500
    Epoch 807/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5292 - binary_accuracy: 0.7500
    Epoch 808/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5291 - binary_accuracy: 0.7500
    Epoch 809/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5289 - binary_accuracy: 0.7500
    Epoch 810/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5287 - binary_accuracy: 0.7500
    Epoch 811/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5286 - binary_accuracy: 0.7500
    Epoch 812/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5284 - binary_accuracy: 0.7500
    Epoch 813/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5282 - binary_accuracy: 0.7500
    Epoch 814/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5280 - binary_accuracy: 0.7500
    Epoch 815/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5279 - binary_accuracy: 0.7500
    Epoch 816/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5277 - binary_accuracy: 0.7500
    Epoch 817/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5275 - binary_accuracy: 0.7500
    Epoch 818/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5274 - binary_accuracy: 0.7500
    Epoch 819/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5272 - binary_accuracy: 0.7500
    Epoch 820/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5270 - binary_accuracy: 0.7500
    Epoch 821/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5268 - binary_accuracy: 0.7500
    Epoch 822/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5267 - binary_accuracy: 0.7500
    Epoch 823/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5265 - binary_accuracy: 0.7500
    Epoch 824/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5263 - binary_accuracy: 0.7500
    Epoch 825/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.5262 - binary_accuracy: 0.7500
    Epoch 826/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5260 - binary_accuracy: 0.7500
    Epoch 827/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.5258 - binary_accuracy: 0.7500
    Epoch 828/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5256 - binary_accuracy: 0.7500
    Epoch 829/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5255 - binary_accuracy: 0.7500
    Epoch 830/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5253 - binary_accuracy: 0.7500
    Epoch 831/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5251 - binary_accuracy: 0.7500
    Epoch 832/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5250 - binary_accuracy: 0.7500
    Epoch 833/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5248 - binary_accuracy: 0.7500
    Epoch 834/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5246 - binary_accuracy: 0.7500
    Epoch 835/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5245 - binary_accuracy: 0.7500
    Epoch 836/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5243 - binary_accuracy: 0.7500
    Epoch 837/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5241 - binary_accuracy: 0.7500
    Epoch 838/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5239 - binary_accuracy: 0.7500
    Epoch 839/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5238 - binary_accuracy: 0.7500
    Epoch 840/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5236 - binary_accuracy: 0.7500
    Epoch 841/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5234 - binary_accuracy: 0.7500
    Epoch 842/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5233 - binary_accuracy: 0.7500
    Epoch 843/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5231 - binary_accuracy: 0.7500
    Epoch 844/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5229 - binary_accuracy: 0.7500
    Epoch 845/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5228 - binary_accuracy: 0.7500
    Epoch 846/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5226 - binary_accuracy: 0.7500
    Epoch 847/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5224 - binary_accuracy: 0.7500
    Epoch 848/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5223 - binary_accuracy: 0.7500
    Epoch 849/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5221 - binary_accuracy: 0.7500
    Epoch 850/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5219 - binary_accuracy: 0.7500
    Epoch 851/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5218 - binary_accuracy: 0.7500
    Epoch 852/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5216 - binary_accuracy: 0.7500
    Epoch 853/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5214 - binary_accuracy: 0.7500
    Epoch 854/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5213 - binary_accuracy: 0.7500
    Epoch 855/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5211 - binary_accuracy: 0.7500
    Epoch 856/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5209 - binary_accuracy: 0.7500
    Epoch 857/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5207 - binary_accuracy: 0.7500
    Epoch 858/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5206 - binary_accuracy: 0.7500
    Epoch 859/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5204 - binary_accuracy: 0.7500
    Epoch 860/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5202 - binary_accuracy: 0.7500
    Epoch 861/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5201 - binary_accuracy: 0.7500
    Epoch 862/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5199 - binary_accuracy: 0.7500
    Epoch 863/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5197 - binary_accuracy: 0.7500
    Epoch 864/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5196 - binary_accuracy: 0.7500
    Epoch 865/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5194 - binary_accuracy: 0.7500
    Epoch 866/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5193 - binary_accuracy: 0.7500
    Epoch 867/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5191 - binary_accuracy: 0.7500
    Epoch 868/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5189 - binary_accuracy: 0.7500
    Epoch 869/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5188 - binary_accuracy: 0.7500
    Epoch 870/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5186 - binary_accuracy: 0.7500
    Epoch 871/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5184 - binary_accuracy: 0.7500
    Epoch 872/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5183 - binary_accuracy: 0.7500
    Epoch 873/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5181 - binary_accuracy: 0.7500
    Epoch 874/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5179 - binary_accuracy: 0.7500
    Epoch 875/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5178 - binary_accuracy: 0.7500
    Epoch 876/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5176 - binary_accuracy: 0.7500
    Epoch 877/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5174 - binary_accuracy: 0.7500
    Epoch 878/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5173 - binary_accuracy: 0.7500
    Epoch 879/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5171 - binary_accuracy: 0.7500
    Epoch 880/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5169 - binary_accuracy: 0.7500
    Epoch 881/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5168 - binary_accuracy: 0.7500
    Epoch 882/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5166 - binary_accuracy: 0.7500
    Epoch 883/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5164 - binary_accuracy: 0.7500
    Epoch 884/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5163 - binary_accuracy: 0.7500
    Epoch 885/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5161 - binary_accuracy: 0.7500
    Epoch 886/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5160 - binary_accuracy: 0.7500
    Epoch 887/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5158 - binary_accuracy: 0.7500
    Epoch 888/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.5156 - binary_accuracy: 0.7500
    Epoch 889/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5155 - binary_accuracy: 0.7500
    Epoch 890/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5153 - binary_accuracy: 0.7500
    Epoch 891/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5151 - binary_accuracy: 0.7500
    Epoch 892/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5150 - binary_accuracy: 0.7500
    Epoch 893/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5148 - binary_accuracy: 0.7500
    Epoch 894/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5146 - binary_accuracy: 0.7500
    Epoch 895/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5145 - binary_accuracy: 0.7500
    Epoch 896/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5143 - binary_accuracy: 0.7500
    Epoch 897/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5142 - binary_accuracy: 0.7500
    Epoch 898/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5140 - binary_accuracy: 0.7500
    Epoch 899/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5138 - binary_accuracy: 0.7500
    Epoch 900/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5137 - binary_accuracy: 0.7500
    Epoch 901/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5135 - binary_accuracy: 0.7500
    Epoch 902/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5133 - binary_accuracy: 0.7500
    Epoch 903/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5132 - binary_accuracy: 0.7500
    Epoch 904/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5130 - binary_accuracy: 0.7500
    Epoch 905/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5129 - binary_accuracy: 0.7500
    Epoch 906/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5127 - binary_accuracy: 0.7500
    Epoch 907/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5125 - binary_accuracy: 0.7500
    Epoch 908/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5124 - binary_accuracy: 0.7500
    Epoch 909/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5122 - binary_accuracy: 0.7500
    Epoch 910/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5120 - binary_accuracy: 0.7500
    Epoch 911/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5119 - binary_accuracy: 0.7500
    Epoch 912/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5117 - binary_accuracy: 0.7500
    Epoch 913/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5116 - binary_accuracy: 0.7500
    Epoch 914/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5114 - binary_accuracy: 0.7500
    Epoch 915/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5112 - binary_accuracy: 0.7500
    Epoch 916/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5111 - binary_accuracy: 0.7500
    Epoch 917/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5109 - binary_accuracy: 0.7500
    Epoch 918/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5108 - binary_accuracy: 0.7500
    Epoch 919/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5106 - binary_accuracy: 0.7500
    Epoch 920/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5104 - binary_accuracy: 0.7500
    Epoch 921/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5103 - binary_accuracy: 0.7500
    Epoch 922/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5101 - binary_accuracy: 0.7500
    Epoch 923/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5100 - binary_accuracy: 0.7500
    Epoch 924/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5098 - binary_accuracy: 0.7500
    Epoch 925/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.5096 - binary_accuracy: 0.7500
    Epoch 926/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5095 - binary_accuracy: 0.7500
    Epoch 927/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5093 - binary_accuracy: 0.7500
    Epoch 928/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5092 - binary_accuracy: 0.7500
    Epoch 929/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.5090 - binary_accuracy: 0.7500
    Epoch 930/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5088 - binary_accuracy: 0.7500
    Epoch 931/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5087 - binary_accuracy: 0.7500
    Epoch 932/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5085 - binary_accuracy: 0.7500
    Epoch 933/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5084 - binary_accuracy: 0.7500
    Epoch 934/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5082 - binary_accuracy: 0.7500
    Epoch 935/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5080 - binary_accuracy: 0.7500
    Epoch 936/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5079 - binary_accuracy: 0.7500
    Epoch 937/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5077 - binary_accuracy: 0.7500
    Epoch 938/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5076 - binary_accuracy: 0.7500
    Epoch 939/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5074 - binary_accuracy: 0.7500
    Epoch 940/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5073 - binary_accuracy: 0.7500
    Epoch 941/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5071 - binary_accuracy: 0.7500
    Epoch 942/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5069 - binary_accuracy: 0.7500
    Epoch 943/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.5068 - binary_accuracy: 0.7500
    Epoch 944/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5066 - binary_accuracy: 0.7500
    Epoch 945/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5065 - binary_accuracy: 0.7500
    Epoch 946/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5063 - binary_accuracy: 0.7500
    Epoch 947/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5062 - binary_accuracy: 0.7500
    Epoch 948/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5060 - binary_accuracy: 0.7500
    Epoch 949/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5058 - binary_accuracy: 0.7500
    Epoch 950/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5057 - binary_accuracy: 0.7500
    Epoch 951/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5055 - binary_accuracy: 0.7500
    Epoch 952/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5054 - binary_accuracy: 0.7500
    Epoch 953/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5052 - binary_accuracy: 0.7500
    Epoch 954/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5051 - binary_accuracy: 0.7500
    Epoch 955/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.5049 - binary_accuracy: 0.7500
    Epoch 956/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5047 - binary_accuracy: 0.7500
    Epoch 957/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5046 - binary_accuracy: 0.7500
    Epoch 958/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.5044 - binary_accuracy: 0.7500
    Epoch 959/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5043 - binary_accuracy: 0.7500
    Epoch 960/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5041 - binary_accuracy: 0.7500
    Epoch 961/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5040 - binary_accuracy: 0.7500
    Epoch 962/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5038 - binary_accuracy: 0.7500
    Epoch 963/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5036 - binary_accuracy: 0.7500
    Epoch 964/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5035 - binary_accuracy: 0.7500
    Epoch 965/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5033 - binary_accuracy: 0.7500
    Epoch 966/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5032 - binary_accuracy: 0.7500
    Epoch 967/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5030 - binary_accuracy: 0.7500
    Epoch 968/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5029 - binary_accuracy: 0.7500
    Epoch 969/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5027 - binary_accuracy: 0.7500
    Epoch 970/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5026 - binary_accuracy: 0.7500
    Epoch 971/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5024 - binary_accuracy: 0.7500
    Epoch 972/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5022 - binary_accuracy: 0.7500
    Epoch 973/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5021 - binary_accuracy: 0.7500
    Epoch 974/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5019 - binary_accuracy: 0.7500
    Epoch 975/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5018 - binary_accuracy: 0.7500
    Epoch 976/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5016 - binary_accuracy: 0.7500
    Epoch 977/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.5015 - binary_accuracy: 0.7500
    Epoch 978/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5013 - binary_accuracy: 0.7500
    Epoch 979/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5012 - binary_accuracy: 0.7500
    Epoch 980/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5010 - binary_accuracy: 0.7500
    Epoch 981/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.5009 - binary_accuracy: 0.7500
    Epoch 982/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5007 - binary_accuracy: 0.7500
    Epoch 983/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5006 - binary_accuracy: 0.7500
    Epoch 984/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5004 - binary_accuracy: 0.7500
    Epoch 985/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.5002 - binary_accuracy: 0.7500
    Epoch 986/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.5001 - binary_accuracy: 0.7500
    Epoch 987/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4999 - binary_accuracy: 0.7500
    Epoch 988/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4998 - binary_accuracy: 0.7500
    Epoch 989/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4996 - binary_accuracy: 0.7500
    Epoch 990/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4995 - binary_accuracy: 0.7500
    Epoch 991/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4993 - binary_accuracy: 0.7500
    Epoch 992/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4992 - binary_accuracy: 0.7500
    Epoch 993/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4990 - binary_accuracy: 0.7500
    Epoch 994/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4989 - binary_accuracy: 0.7500
    Epoch 995/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4987 - binary_accuracy: 0.7500
    Epoch 996/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4986 - binary_accuracy: 0.7500
    Epoch 997/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4984 - binary_accuracy: 0.7500
    Epoch 998/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4983 - binary_accuracy: 0.7500
    Epoch 999/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4981 - binary_accuracy: 0.7500
    Epoch 1000/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4980 - binary_accuracy: 0.7500
    Epoch 1001/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4978 - binary_accuracy: 0.7500
    Epoch 1002/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4976 - binary_accuracy: 0.7500
    Epoch 1003/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4975 - binary_accuracy: 0.7500
    Epoch 1004/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4973 - binary_accuracy: 0.7500
    Epoch 1005/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4972 - binary_accuracy: 0.7500
    Epoch 1006/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4970 - binary_accuracy: 0.7500
    Epoch 1007/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4969 - binary_accuracy: 0.7500
    Epoch 1008/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4967 - binary_accuracy: 0.7500
    Epoch 1009/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4966 - binary_accuracy: 0.7500
    Epoch 1010/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4964 - binary_accuracy: 0.7500
    Epoch 1011/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4963 - binary_accuracy: 0.7500
    Epoch 1012/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.4961 - binary_accuracy: 0.7500
    Epoch 1013/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4960 - binary_accuracy: 0.7500
    Epoch 1014/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4958 - binary_accuracy: 0.7500
    Epoch 1015/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4957 - binary_accuracy: 0.7500
    Epoch 1016/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4955 - binary_accuracy: 0.7500
    Epoch 1017/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4954 - binary_accuracy: 0.7500
    Epoch 1018/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4952 - binary_accuracy: 0.7500
    Epoch 1019/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4951 - binary_accuracy: 0.7500
    Epoch 1020/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4949 - binary_accuracy: 0.7500
    Epoch 1021/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4948 - binary_accuracy: 0.7500
    Epoch 1022/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4946 - binary_accuracy: 0.7500
    Epoch 1023/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4945 - binary_accuracy: 0.7500
    Epoch 1024/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4943 - binary_accuracy: 0.7500
    Epoch 1025/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4942 - binary_accuracy: 0.7500
    Epoch 1026/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4940 - binary_accuracy: 0.7500
    Epoch 1027/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4939 - binary_accuracy: 0.7500
    Epoch 1028/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4937 - binary_accuracy: 0.7500
    Epoch 1029/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4936 - binary_accuracy: 0.7500
    Epoch 1030/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4934 - binary_accuracy: 0.7500
    Epoch 1031/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4933 - binary_accuracy: 0.7500
    Epoch 1032/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4931 - binary_accuracy: 0.7500
    Epoch 1033/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4930 - binary_accuracy: 0.7500
    Epoch 1034/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4928 - binary_accuracy: 0.7500
    Epoch 1035/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4927 - binary_accuracy: 0.7500
    Epoch 1036/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4925 - binary_accuracy: 0.7500
    Epoch 1037/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4924 - binary_accuracy: 0.7500
    Epoch 1038/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4922 - binary_accuracy: 0.7500
    Epoch 1039/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4921 - binary_accuracy: 0.7500
    Epoch 1040/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4919 - binary_accuracy: 0.7500
    Epoch 1041/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4918 - binary_accuracy: 0.7500
    Epoch 1042/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4917 - binary_accuracy: 0.7500
    Epoch 1043/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4915 - binary_accuracy: 0.7500
    Epoch 1044/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4914 - binary_accuracy: 0.7500
    Epoch 1045/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4912 - binary_accuracy: 0.7500
    Epoch 1046/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4911 - binary_accuracy: 0.7500
    Epoch 1047/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4909 - binary_accuracy: 0.7500
    Epoch 1048/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4908 - binary_accuracy: 0.7500
    Epoch 1049/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4906 - binary_accuracy: 0.7500
    Epoch 1050/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4905 - binary_accuracy: 0.7500
    Epoch 1051/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4903 - binary_accuracy: 0.7500
    Epoch 1052/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4902 - binary_accuracy: 0.7500
    Epoch 1053/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4900 - binary_accuracy: 0.7500
    Epoch 1054/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4899 - binary_accuracy: 0.7500
    Epoch 1055/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4897 - binary_accuracy: 0.7500
    Epoch 1056/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4896 - binary_accuracy: 0.7500
    Epoch 1057/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4894 - binary_accuracy: 0.7500
    Epoch 1058/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4893 - binary_accuracy: 0.7500
    Epoch 1059/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4892 - binary_accuracy: 0.7500
    Epoch 1060/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.4890 - binary_accuracy: 0.7500
    Epoch 1061/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.4889 - binary_accuracy: 0.7500
    Epoch 1062/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.4887 - binary_accuracy: 0.7500
    Epoch 1063/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4886 - binary_accuracy: 0.7500
    Epoch 1064/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4884 - binary_accuracy: 0.7500
    Epoch 1065/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4883 - binary_accuracy: 0.7500
    Epoch 1066/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4881 - binary_accuracy: 0.7500
    Epoch 1067/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4880 - binary_accuracy: 0.7500
    Epoch 1068/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4878 - binary_accuracy: 0.7500
    Epoch 1069/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4877 - binary_accuracy: 0.7500
    Epoch 1070/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4875 - binary_accuracy: 0.7500
    Epoch 1071/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4874 - binary_accuracy: 0.7500
    Epoch 1072/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4873 - binary_accuracy: 0.7500
    Epoch 1073/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4871 - binary_accuracy: 0.7500
    Epoch 1074/7000
    1/1 [==============================] - 0s 39ms/step - loss: 0.4870 - binary_accuracy: 0.7500
    Epoch 1075/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4868 - binary_accuracy: 0.7500
    Epoch 1076/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4867 - binary_accuracy: 0.7500
    Epoch 1077/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4865 - binary_accuracy: 0.7500
    Epoch 1078/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4864 - binary_accuracy: 0.7500
    Epoch 1079/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4862 - binary_accuracy: 0.7500
    Epoch 1080/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4861 - binary_accuracy: 0.7500
    Epoch 1081/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4860 - binary_accuracy: 0.7500
    Epoch 1082/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4858 - binary_accuracy: 0.7500
    Epoch 1083/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4857 - binary_accuracy: 0.7500
    Epoch 1084/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4855 - binary_accuracy: 0.7500
    Epoch 1085/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4854 - binary_accuracy: 0.7500
    Epoch 1086/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4852 - binary_accuracy: 0.7500
    Epoch 1087/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4851 - binary_accuracy: 0.7500
    Epoch 1088/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4849 - binary_accuracy: 0.7500
    Epoch 1089/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4848 - binary_accuracy: 0.7500
    Epoch 1090/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4847 - binary_accuracy: 0.7500
    Epoch 1091/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.4845 - binary_accuracy: 0.7500
    Epoch 1092/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4844 - binary_accuracy: 0.7500
    Epoch 1093/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4842 - binary_accuracy: 0.7500
    Epoch 1094/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4841 - binary_accuracy: 0.7500
    Epoch 1095/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4839 - binary_accuracy: 0.7500
    Epoch 1096/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.4838 - binary_accuracy: 0.7500
    Epoch 1097/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4837 - binary_accuracy: 0.7500
    Epoch 1098/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4835 - binary_accuracy: 0.7500
    Epoch 1099/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4834 - binary_accuracy: 0.7500
    Epoch 1100/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4832 - binary_accuracy: 0.7500
    Epoch 1101/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4831 - binary_accuracy: 0.7500
    Epoch 1102/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4829 - binary_accuracy: 0.7500
    Epoch 1103/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4828 - binary_accuracy: 0.7500
    Epoch 1104/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4827 - binary_accuracy: 0.7500
    Epoch 1105/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4825 - binary_accuracy: 0.7500
    Epoch 1106/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4824 - binary_accuracy: 0.7500
    Epoch 1107/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4822 - binary_accuracy: 0.7500
    Epoch 1108/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4821 - binary_accuracy: 0.7500
    Epoch 1109/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4819 - binary_accuracy: 0.7500
    Epoch 1110/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4818 - binary_accuracy: 0.7500
    Epoch 1111/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4817 - binary_accuracy: 0.7500
    Epoch 1112/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4815 - binary_accuracy: 0.7500
    Epoch 1113/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4814 - binary_accuracy: 0.7500
    Epoch 1114/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4812 - binary_accuracy: 0.7500
    Epoch 1115/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.4811 - binary_accuracy: 0.7500
    Epoch 1116/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4810 - binary_accuracy: 0.7500
    Epoch 1117/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4808 - binary_accuracy: 0.7500
    Epoch 1118/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4807 - binary_accuracy: 0.7500
    Epoch 1119/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4805 - binary_accuracy: 0.7500
    Epoch 1120/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4804 - binary_accuracy: 0.7500
    Epoch 1121/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4803 - binary_accuracy: 0.7500
    Epoch 1122/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4801 - binary_accuracy: 0.7500
    Epoch 1123/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4800 - binary_accuracy: 0.7500
    Epoch 1124/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4798 - binary_accuracy: 0.7500
    Epoch 1125/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4797 - binary_accuracy: 0.7500
    Epoch 1126/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4796 - binary_accuracy: 0.7500
    Epoch 1127/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4794 - binary_accuracy: 0.7500
    Epoch 1128/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4793 - binary_accuracy: 0.7500
    Epoch 1129/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4791 - binary_accuracy: 0.7500
    Epoch 1130/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4790 - binary_accuracy: 0.7500
    Epoch 1131/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4789 - binary_accuracy: 0.7500
    Epoch 1132/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4787 - binary_accuracy: 0.7500
    Epoch 1133/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4786 - binary_accuracy: 0.7500
    Epoch 1134/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4784 - binary_accuracy: 0.7500
    Epoch 1135/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.4783 - binary_accuracy: 0.7500
    Epoch 1136/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4782 - binary_accuracy: 0.7500
    Epoch 1137/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4780 - binary_accuracy: 0.7500
    Epoch 1138/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4779 - binary_accuracy: 0.7500
    Epoch 1139/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4777 - binary_accuracy: 0.7500
    Epoch 1140/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4776 - binary_accuracy: 0.7500
    Epoch 1141/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4775 - binary_accuracy: 0.7500
    Epoch 1142/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4773 - binary_accuracy: 0.7500
    Epoch 1143/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4772 - binary_accuracy: 0.7500
    Epoch 1144/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4770 - binary_accuracy: 0.7500
    Epoch 1145/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4769 - binary_accuracy: 0.7500
    Epoch 1146/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4768 - binary_accuracy: 0.7500
    Epoch 1147/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4766 - binary_accuracy: 0.7500
    Epoch 1148/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4765 - binary_accuracy: 0.7500
    Epoch 1149/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4763 - binary_accuracy: 0.7500
    Epoch 1150/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4762 - binary_accuracy: 0.7500
    Epoch 1151/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4761 - binary_accuracy: 0.7500
    Epoch 1152/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4759 - binary_accuracy: 0.7500
    Epoch 1153/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4758 - binary_accuracy: 0.7500
    Epoch 1154/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4757 - binary_accuracy: 0.7500
    Epoch 1155/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4755 - binary_accuracy: 0.7500
    Epoch 1156/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4754 - binary_accuracy: 0.7500
    Epoch 1157/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4752 - binary_accuracy: 0.7500
    Epoch 1158/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4751 - binary_accuracy: 0.7500
    Epoch 1159/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4750 - binary_accuracy: 0.7500
    Epoch 1160/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4748 - binary_accuracy: 0.7500
    Epoch 1161/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4747 - binary_accuracy: 0.7500
    Epoch 1162/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4746 - binary_accuracy: 0.7500
    Epoch 1163/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4744 - binary_accuracy: 0.7500
    Epoch 1164/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4743 - binary_accuracy: 0.7500
    Epoch 1165/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4741 - binary_accuracy: 0.7500
    Epoch 1166/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4740 - binary_accuracy: 0.7500
    Epoch 1167/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4739 - binary_accuracy: 0.7500
    Epoch 1168/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4737 - binary_accuracy: 0.7500
    Epoch 1169/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4736 - binary_accuracy: 0.7500
    Epoch 1170/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4735 - binary_accuracy: 0.7500
    Epoch 1171/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4733 - binary_accuracy: 0.7500
    Epoch 1172/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4732 - binary_accuracy: 0.7500
    Epoch 1173/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4731 - binary_accuracy: 0.7500
    Epoch 1174/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4729 - binary_accuracy: 0.7500
    Epoch 1175/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4728 - binary_accuracy: 0.7500
    Epoch 1176/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4726 - binary_accuracy: 0.7500
    Epoch 1177/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4725 - binary_accuracy: 0.7500
    Epoch 1178/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4724 - binary_accuracy: 0.7500
    Epoch 1179/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4722 - binary_accuracy: 0.7500
    Epoch 1180/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4721 - binary_accuracy: 0.7500
    Epoch 1181/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4720 - binary_accuracy: 0.7500
    Epoch 1182/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4718 - binary_accuracy: 0.7500
    Epoch 1183/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4717 - binary_accuracy: 0.7500
    Epoch 1184/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4716 - binary_accuracy: 0.7500
    Epoch 1185/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4714 - binary_accuracy: 0.7500
    Epoch 1186/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4713 - binary_accuracy: 0.7500
    Epoch 1187/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4712 - binary_accuracy: 0.7500
    Epoch 1188/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.4710 - binary_accuracy: 0.7500
    Epoch 1189/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4709 - binary_accuracy: 0.7500
    Epoch 1190/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4708 - binary_accuracy: 0.7500
    Epoch 1191/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4706 - binary_accuracy: 0.7500
    Epoch 1192/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4705 - binary_accuracy: 0.7500
    Epoch 1193/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4703 - binary_accuracy: 0.7500
    Epoch 1194/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4702 - binary_accuracy: 0.7500
    Epoch 1195/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4701 - binary_accuracy: 0.7500
    Epoch 1196/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4699 - binary_accuracy: 0.7500
    Epoch 1197/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4698 - binary_accuracy: 0.7500
    Epoch 1198/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4697 - binary_accuracy: 0.7500
    Epoch 1199/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4695 - binary_accuracy: 0.7500
    Epoch 1200/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4694 - binary_accuracy: 0.7500
    Epoch 1201/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4693 - binary_accuracy: 0.7500
    Epoch 1202/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4691 - binary_accuracy: 0.7500
    Epoch 1203/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4690 - binary_accuracy: 0.7500
    Epoch 1204/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4689 - binary_accuracy: 0.7500
    Epoch 1205/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4687 - binary_accuracy: 0.7500
    Epoch 1206/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4686 - binary_accuracy: 0.7500
    Epoch 1207/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4685 - binary_accuracy: 0.7500
    Epoch 1208/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4683 - binary_accuracy: 0.7500
    Epoch 1209/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4682 - binary_accuracy: 0.7500
    Epoch 1210/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4681 - binary_accuracy: 0.7500
    Epoch 1211/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4679 - binary_accuracy: 0.7500
    Epoch 1212/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4678 - binary_accuracy: 0.7500
    Epoch 1213/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4677 - binary_accuracy: 0.7500
    Epoch 1214/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4675 - binary_accuracy: 0.7500
    Epoch 1215/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4674 - binary_accuracy: 0.7500
    Epoch 1216/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4673 - binary_accuracy: 0.7500
    Epoch 1217/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4671 - binary_accuracy: 0.7500
    Epoch 1218/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4670 - binary_accuracy: 0.7500
    Epoch 1219/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4669 - binary_accuracy: 0.7500
    Epoch 1220/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4667 - binary_accuracy: 0.7500
    Epoch 1221/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4666 - binary_accuracy: 0.7500
    Epoch 1222/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4665 - binary_accuracy: 0.7500
    Epoch 1223/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4664 - binary_accuracy: 0.7500
    Epoch 1224/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4662 - binary_accuracy: 0.7500
    Epoch 1225/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4661 - binary_accuracy: 0.7500
    Epoch 1226/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4660 - binary_accuracy: 0.7500
    Epoch 1227/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4658 - binary_accuracy: 0.7500
    Epoch 1228/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4657 - binary_accuracy: 0.7500
    Epoch 1229/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4656 - binary_accuracy: 0.7500
    Epoch 1230/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4654 - binary_accuracy: 0.7500
    Epoch 1231/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4653 - binary_accuracy: 0.7500
    Epoch 1232/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4652 - binary_accuracy: 0.7500
    Epoch 1233/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4650 - binary_accuracy: 0.7500
    Epoch 1234/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4649 - binary_accuracy: 0.7500
    Epoch 1235/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4648 - binary_accuracy: 0.7500
    Epoch 1236/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4646 - binary_accuracy: 0.7500
    Epoch 1237/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4645 - binary_accuracy: 0.7500
    Epoch 1238/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4644 - binary_accuracy: 0.7500
    Epoch 1239/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4642 - binary_accuracy: 0.7500
    Epoch 1240/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4641 - binary_accuracy: 0.7500
    Epoch 1241/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4640 - binary_accuracy: 0.7500
    Epoch 1242/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4639 - binary_accuracy: 0.7500
    Epoch 1243/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4637 - binary_accuracy: 0.7500
    Epoch 1244/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4636 - binary_accuracy: 0.7500
    Epoch 1245/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.4635 - binary_accuracy: 0.7500
    Epoch 1246/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4633 - binary_accuracy: 0.7500
    Epoch 1247/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4632 - binary_accuracy: 0.7500
    Epoch 1248/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4631 - binary_accuracy: 0.7500
    Epoch 1249/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4629 - binary_accuracy: 0.7500
    Epoch 1250/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4628 - binary_accuracy: 0.7500
    Epoch 1251/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4627 - binary_accuracy: 0.7500
    Epoch 1252/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4626 - binary_accuracy: 0.7500
    Epoch 1253/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4624 - binary_accuracy: 0.7500
    Epoch 1254/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4623 - binary_accuracy: 0.7500
    Epoch 1255/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4622 - binary_accuracy: 0.7500
    Epoch 1256/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4620 - binary_accuracy: 0.7500
    Epoch 1257/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4619 - binary_accuracy: 0.7500
    Epoch 1258/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4618 - binary_accuracy: 0.7500
    Epoch 1259/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4616 - binary_accuracy: 0.7500
    Epoch 1260/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4615 - binary_accuracy: 0.7500
    Epoch 1261/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4614 - binary_accuracy: 0.7500
    Epoch 1262/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4613 - binary_accuracy: 0.7500
    Epoch 1263/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4611 - binary_accuracy: 0.7500
    Epoch 1264/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4610 - binary_accuracy: 0.7500
    Epoch 1265/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4609 - binary_accuracy: 0.7500
    Epoch 1266/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4607 - binary_accuracy: 0.7500
    Epoch 1267/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4606 - binary_accuracy: 0.7500
    Epoch 1268/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4605 - binary_accuracy: 0.7500
    Epoch 1269/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4604 - binary_accuracy: 0.7500
    Epoch 1270/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4602 - binary_accuracy: 0.7500
    Epoch 1271/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4601 - binary_accuracy: 0.7500
    Epoch 1272/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4600 - binary_accuracy: 0.7500
    Epoch 1273/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4598 - binary_accuracy: 0.7500
    Epoch 1274/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4597 - binary_accuracy: 0.7500
    Epoch 1275/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4596 - binary_accuracy: 0.7500
    Epoch 1276/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4595 - binary_accuracy: 0.7500
    Epoch 1277/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4593 - binary_accuracy: 0.7500
    Epoch 1278/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4592 - binary_accuracy: 0.7500
    Epoch 1279/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4591 - binary_accuracy: 0.7500
    Epoch 1280/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.4590 - binary_accuracy: 0.7500
    Epoch 1281/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.4588 - binary_accuracy: 0.7500
    Epoch 1282/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.4587 - binary_accuracy: 0.7500
    Epoch 1283/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4586 - binary_accuracy: 0.7500
    Epoch 1284/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4584 - binary_accuracy: 0.7500
    Epoch 1285/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4583 - binary_accuracy: 0.7500
    Epoch 1286/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4582 - binary_accuracy: 0.7500
    Epoch 1287/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4581 - binary_accuracy: 0.7500
    Epoch 1288/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4579 - binary_accuracy: 0.7500
    Epoch 1289/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4578 - binary_accuracy: 0.7500
    Epoch 1290/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4577 - binary_accuracy: 0.7500
    Epoch 1291/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4576 - binary_accuracy: 0.7500
    Epoch 1292/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4574 - binary_accuracy: 0.7500
    Epoch 1293/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4573 - binary_accuracy: 0.7500
    Epoch 1294/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4572 - binary_accuracy: 0.7500
    Epoch 1295/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4571 - binary_accuracy: 0.7500
    Epoch 1296/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4569 - binary_accuracy: 0.7500
    Epoch 1297/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4568 - binary_accuracy: 0.7500
    Epoch 1298/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4567 - binary_accuracy: 0.7500
    Epoch 1299/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4565 - binary_accuracy: 0.7500
    Epoch 1300/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4564 - binary_accuracy: 0.7500
    Epoch 1301/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4563 - binary_accuracy: 0.7500
    Epoch 1302/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4562 - binary_accuracy: 0.7500
    Epoch 1303/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4560 - binary_accuracy: 0.7500
    Epoch 1304/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4559 - binary_accuracy: 0.7500
    Epoch 1305/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4558 - binary_accuracy: 0.7500
    Epoch 1306/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4557 - binary_accuracy: 0.7500
    Epoch 1307/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4555 - binary_accuracy: 0.7500
    Epoch 1308/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4554 - binary_accuracy: 0.7500
    Epoch 1309/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4553 - binary_accuracy: 0.7500
    Epoch 1310/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4552 - binary_accuracy: 0.7500
    Epoch 1311/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4550 - binary_accuracy: 0.7500
    Epoch 1312/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4549 - binary_accuracy: 0.7500
    Epoch 1313/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4548 - binary_accuracy: 0.7500
    Epoch 1314/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4547 - binary_accuracy: 0.7500
    Epoch 1315/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4545 - binary_accuracy: 0.7500
    Epoch 1316/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4544 - binary_accuracy: 0.7500
    Epoch 1317/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4543 - binary_accuracy: 0.7500
    Epoch 1318/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4542 - binary_accuracy: 0.7500
    Epoch 1319/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4540 - binary_accuracy: 0.7500
    Epoch 1320/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.4539 - binary_accuracy: 0.7500
    Epoch 1321/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4538 - binary_accuracy: 0.7500
    Epoch 1322/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4537 - binary_accuracy: 0.7500
    Epoch 1323/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4535 - binary_accuracy: 0.7500
    Epoch 1324/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4534 - binary_accuracy: 0.7500
    Epoch 1325/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4533 - binary_accuracy: 0.7500
    Epoch 1326/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4532 - binary_accuracy: 0.7500
    Epoch 1327/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4530 - binary_accuracy: 0.7500
    Epoch 1328/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4529 - binary_accuracy: 0.7500
    Epoch 1329/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4528 - binary_accuracy: 0.7500
    Epoch 1330/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4527 - binary_accuracy: 0.7500
    Epoch 1331/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4526 - binary_accuracy: 0.7500
    Epoch 1332/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4524 - binary_accuracy: 0.7500
    Epoch 1333/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4523 - binary_accuracy: 0.7500
    Epoch 1334/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4522 - binary_accuracy: 0.7500
    Epoch 1335/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4521 - binary_accuracy: 0.7500
    Epoch 1336/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4519 - binary_accuracy: 0.7500
    Epoch 1337/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4518 - binary_accuracy: 0.7500
    Epoch 1338/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4517 - binary_accuracy: 0.7500
    Epoch 1339/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4516 - binary_accuracy: 0.7500
    Epoch 1340/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4514 - binary_accuracy: 0.7500
    Epoch 1341/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4513 - binary_accuracy: 0.7500
    Epoch 1342/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4512 - binary_accuracy: 0.7500
    Epoch 1343/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4511 - binary_accuracy: 0.7500
    Epoch 1344/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4510 - binary_accuracy: 0.7500
    Epoch 1345/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4508 - binary_accuracy: 0.7500
    Epoch 1346/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4507 - binary_accuracy: 0.7500
    Epoch 1347/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4506 - binary_accuracy: 0.7500
    Epoch 1348/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4505 - binary_accuracy: 0.7500
    Epoch 1349/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4503 - binary_accuracy: 0.7500
    Epoch 1350/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4502 - binary_accuracy: 0.7500
    Epoch 1351/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4501 - binary_accuracy: 0.7500
    Epoch 1352/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4500 - binary_accuracy: 0.7500
    Epoch 1353/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4498 - binary_accuracy: 0.7500
    Epoch 1354/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4497 - binary_accuracy: 0.7500
    Epoch 1355/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4496 - binary_accuracy: 0.7500
    Epoch 1356/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4495 - binary_accuracy: 0.7500
    Epoch 1357/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4494 - binary_accuracy: 0.7500
    Epoch 1358/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4492 - binary_accuracy: 0.7500
    Epoch 1359/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4491 - binary_accuracy: 0.7500
    Epoch 1360/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4490 - binary_accuracy: 0.7500
    Epoch 1361/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4489 - binary_accuracy: 0.7500
    Epoch 1362/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4488 - binary_accuracy: 0.7500
    Epoch 1363/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.4486 - binary_accuracy: 0.7500
    Epoch 1364/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4485 - binary_accuracy: 0.7500
    Epoch 1365/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4484 - binary_accuracy: 0.7500
    Epoch 1366/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4483 - binary_accuracy: 0.7500
    Epoch 1367/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4481 - binary_accuracy: 0.7500
    Epoch 1368/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4480 - binary_accuracy: 0.7500
    Epoch 1369/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4479 - binary_accuracy: 0.7500
    Epoch 1370/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4478 - binary_accuracy: 0.7500
    Epoch 1371/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4477 - binary_accuracy: 0.7500
    Epoch 1372/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4475 - binary_accuracy: 0.7500
    Epoch 1373/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4474 - binary_accuracy: 0.7500
    Epoch 1374/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4473 - binary_accuracy: 0.7500
    Epoch 1375/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4472 - binary_accuracy: 0.7500
    Epoch 1376/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4471 - binary_accuracy: 0.7500
    Epoch 1377/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4469 - binary_accuracy: 0.7500
    Epoch 1378/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4468 - binary_accuracy: 0.7500
    Epoch 1379/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4467 - binary_accuracy: 0.7500
    Epoch 1380/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4466 - binary_accuracy: 0.7500
    Epoch 1381/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4465 - binary_accuracy: 0.7500
    Epoch 1382/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4463 - binary_accuracy: 0.7500
    Epoch 1383/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4462 - binary_accuracy: 0.7500
    Epoch 1384/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4461 - binary_accuracy: 0.7500
    Epoch 1385/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4460 - binary_accuracy: 0.7500
    Epoch 1386/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4459 - binary_accuracy: 0.7500
    Epoch 1387/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4457 - binary_accuracy: 0.7500
    Epoch 1388/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4456 - binary_accuracy: 0.7500
    Epoch 1389/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4455 - binary_accuracy: 0.7500
    Epoch 1390/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4454 - binary_accuracy: 0.7500
    Epoch 1391/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4453 - binary_accuracy: 0.7500
    Epoch 1392/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4451 - binary_accuracy: 0.7500
    Epoch 1393/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4450 - binary_accuracy: 0.7500
    Epoch 1394/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4449 - binary_accuracy: 0.7500
    Epoch 1395/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4448 - binary_accuracy: 0.7500
    Epoch 1396/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4447 - binary_accuracy: 0.7500
    Epoch 1397/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4446 - binary_accuracy: 0.7500
    Epoch 1398/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4444 - binary_accuracy: 0.7500
    Epoch 1399/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4443 - binary_accuracy: 0.7500
    Epoch 1400/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4442 - binary_accuracy: 0.7500
    Epoch 1401/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4441 - binary_accuracy: 0.7500
    Epoch 1402/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4440 - binary_accuracy: 0.7500
    Epoch 1403/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4438 - binary_accuracy: 0.7500
    Epoch 1404/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4437 - binary_accuracy: 0.7500
    Epoch 1405/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4436 - binary_accuracy: 0.7500
    Epoch 1406/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4435 - binary_accuracy: 0.7500
    Epoch 1407/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4434 - binary_accuracy: 0.7500
    Epoch 1408/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4432 - binary_accuracy: 0.7500
    Epoch 1409/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4431 - binary_accuracy: 0.7500
    Epoch 1410/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4430 - binary_accuracy: 0.7500
    Epoch 1411/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4429 - binary_accuracy: 0.7500
    Epoch 1412/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4428 - binary_accuracy: 0.7500
    Epoch 1413/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4427 - binary_accuracy: 0.7500
    Epoch 1414/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4425 - binary_accuracy: 0.7500
    Epoch 1415/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4424 - binary_accuracy: 0.7500
    Epoch 1416/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4423 - binary_accuracy: 0.7500
    Epoch 1417/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4422 - binary_accuracy: 0.7500
    Epoch 1418/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4421 - binary_accuracy: 0.7500
    Epoch 1419/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4420 - binary_accuracy: 0.7500
    Epoch 1420/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4418 - binary_accuracy: 0.7500
    Epoch 1421/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4417 - binary_accuracy: 0.7500
    Epoch 1422/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4416 - binary_accuracy: 0.7500
    Epoch 1423/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4415 - binary_accuracy: 0.7500
    Epoch 1424/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4414 - binary_accuracy: 0.7500
    Epoch 1425/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4412 - binary_accuracy: 0.7500
    Epoch 1426/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4411 - binary_accuracy: 0.7500
    Epoch 1427/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4410 - binary_accuracy: 0.7500
    Epoch 1428/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4409 - binary_accuracy: 0.7500
    Epoch 1429/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4408 - binary_accuracy: 0.7500
    Epoch 1430/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4407 - binary_accuracy: 0.7500
    Epoch 1431/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4405 - binary_accuracy: 0.7500
    Epoch 1432/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4404 - binary_accuracy: 0.7500
    Epoch 1433/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4403 - binary_accuracy: 0.7500
    Epoch 1434/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4402 - binary_accuracy: 0.7500
    Epoch 1435/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4401 - binary_accuracy: 0.7500
    Epoch 1436/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4400 - binary_accuracy: 0.7500
    Epoch 1437/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4399 - binary_accuracy: 0.7500
    Epoch 1438/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4397 - binary_accuracy: 0.7500
    Epoch 1439/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4396 - binary_accuracy: 0.7500
    Epoch 1440/7000
    1/1 [==============================] - 0s 22ms/step - loss: 0.4395 - binary_accuracy: 0.7500
    Epoch 1441/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4394 - binary_accuracy: 0.7500
    Epoch 1442/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4393 - binary_accuracy: 0.7500
    Epoch 1443/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4392 - binary_accuracy: 0.7500
    Epoch 1444/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4390 - binary_accuracy: 0.7500
    Epoch 1445/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4389 - binary_accuracy: 0.7500
    Epoch 1446/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4388 - binary_accuracy: 0.7500
    Epoch 1447/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4387 - binary_accuracy: 0.7500
    Epoch 1448/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4386 - binary_accuracy: 0.7500
    Epoch 1449/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4385 - binary_accuracy: 0.7500
    Epoch 1450/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4383 - binary_accuracy: 0.7500
    Epoch 1451/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4382 - binary_accuracy: 0.7500
    Epoch 1452/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4381 - binary_accuracy: 0.7500
    Epoch 1453/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4380 - binary_accuracy: 0.7500
    Epoch 1454/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4379 - binary_accuracy: 0.7500
    Epoch 1455/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4378 - binary_accuracy: 0.7500
    Epoch 1456/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4377 - binary_accuracy: 0.7500
    Epoch 1457/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4375 - binary_accuracy: 0.7500
    Epoch 1458/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4374 - binary_accuracy: 0.7500
    Epoch 1459/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4373 - binary_accuracy: 0.7500
    Epoch 1460/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4372 - binary_accuracy: 0.7500
    Epoch 1461/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4371 - binary_accuracy: 0.7500
    Epoch 1462/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4370 - binary_accuracy: 0.7500
    Epoch 1463/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4369 - binary_accuracy: 0.7500
    Epoch 1464/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4367 - binary_accuracy: 0.7500
    Epoch 1465/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4366 - binary_accuracy: 0.7500
    Epoch 1466/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4365 - binary_accuracy: 0.7500
    Epoch 1467/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4364 - binary_accuracy: 0.7500
    Epoch 1468/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4363 - binary_accuracy: 0.7500
    Epoch 1469/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4362 - binary_accuracy: 0.7500
    Epoch 1470/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4361 - binary_accuracy: 0.7500
    Epoch 1471/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4359 - binary_accuracy: 0.7500
    Epoch 1472/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4358 - binary_accuracy: 0.7500
    Epoch 1473/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4357 - binary_accuracy: 0.7500
    Epoch 1474/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4356 - binary_accuracy: 0.7500
    Epoch 1475/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4355 - binary_accuracy: 0.7500
    Epoch 1476/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4354 - binary_accuracy: 0.7500
    Epoch 1477/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4353 - binary_accuracy: 0.7500
    Epoch 1478/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4351 - binary_accuracy: 0.7500
    Epoch 1479/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4350 - binary_accuracy: 0.7500
    Epoch 1480/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.4349 - binary_accuracy: 0.7500
    Epoch 1481/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4348 - binary_accuracy: 0.7500
    Epoch 1482/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4347 - binary_accuracy: 0.7500
    Epoch 1483/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4346 - binary_accuracy: 0.7500
    Epoch 1484/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4345 - binary_accuracy: 0.7500
    Epoch 1485/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4344 - binary_accuracy: 0.7500
    Epoch 1486/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4342 - binary_accuracy: 0.7500
    Epoch 1487/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4341 - binary_accuracy: 0.7500
    Epoch 1488/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4340 - binary_accuracy: 0.7500
    Epoch 1489/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4339 - binary_accuracy: 0.7500
    Epoch 1490/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4338 - binary_accuracy: 0.7500
    Epoch 1491/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4337 - binary_accuracy: 0.7500
    Epoch 1492/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4336 - binary_accuracy: 0.7500
    Epoch 1493/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4335 - binary_accuracy: 0.7500
    Epoch 1494/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4333 - binary_accuracy: 0.7500
    Epoch 1495/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4332 - binary_accuracy: 0.7500
    Epoch 1496/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4331 - binary_accuracy: 0.7500
    Epoch 1497/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4330 - binary_accuracy: 0.7500
    Epoch 1498/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4329 - binary_accuracy: 0.7500
    Epoch 1499/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4328 - binary_accuracy: 0.7500
    Epoch 1500/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4327 - binary_accuracy: 0.7500
    Epoch 1501/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4326 - binary_accuracy: 0.7500
    Epoch 1502/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4324 - binary_accuracy: 0.7500
    Epoch 1503/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4323 - binary_accuracy: 0.7500
    Epoch 1504/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4322 - binary_accuracy: 0.7500
    Epoch 1505/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4321 - binary_accuracy: 0.7500
    Epoch 1506/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4320 - binary_accuracy: 0.7500
    Epoch 1507/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4319 - binary_accuracy: 0.7500
    Epoch 1508/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4318 - binary_accuracy: 0.7500
    Epoch 1509/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4317 - binary_accuracy: 0.7500
    Epoch 1510/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4316 - binary_accuracy: 0.7500
    Epoch 1511/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4314 - binary_accuracy: 0.7500
    Epoch 1512/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4313 - binary_accuracy: 0.7500
    Epoch 1513/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4312 - binary_accuracy: 0.7500
    Epoch 1514/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4311 - binary_accuracy: 0.7500
    Epoch 1515/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4310 - binary_accuracy: 0.7500
    Epoch 1516/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4309 - binary_accuracy: 0.7500
    Epoch 1517/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4308 - binary_accuracy: 0.7500
    Epoch 1518/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4307 - binary_accuracy: 0.7500
    Epoch 1519/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4306 - binary_accuracy: 0.7500
    Epoch 1520/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4304 - binary_accuracy: 0.7500
    Epoch 1521/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4303 - binary_accuracy: 0.7500
    Epoch 1522/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4302 - binary_accuracy: 0.7500
    Epoch 1523/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4301 - binary_accuracy: 0.7500
    Epoch 1524/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4300 - binary_accuracy: 0.7500
    Epoch 1525/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4299 - binary_accuracy: 0.7500
    Epoch 1526/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4298 - binary_accuracy: 0.7500
    Epoch 1527/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4297 - binary_accuracy: 0.7500
    Epoch 1528/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4296 - binary_accuracy: 0.7500
    Epoch 1529/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4294 - binary_accuracy: 0.7500
    Epoch 1530/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4293 - binary_accuracy: 0.7500
    Epoch 1531/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4292 - binary_accuracy: 0.7500
    Epoch 1532/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4291 - binary_accuracy: 0.7500
    Epoch 1533/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4290 - binary_accuracy: 0.7500
    Epoch 1534/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4289 - binary_accuracy: 0.7500
    Epoch 1535/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4288 - binary_accuracy: 0.7500
    Epoch 1536/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.4287 - binary_accuracy: 0.7500
    Epoch 1537/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4286 - binary_accuracy: 0.7500
    Epoch 1538/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4285 - binary_accuracy: 0.7500
    Epoch 1539/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4283 - binary_accuracy: 0.7500
    Epoch 1540/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4282 - binary_accuracy: 0.7500
    Epoch 1541/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4281 - binary_accuracy: 0.7500
    Epoch 1542/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4280 - binary_accuracy: 0.7500
    Epoch 1543/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4279 - binary_accuracy: 0.7500
    Epoch 1544/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4278 - binary_accuracy: 0.7500
    Epoch 1545/7000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4277 - binary_accuracy: 0.7500
    Epoch 1546/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4276 - binary_accuracy: 0.7500
    Epoch 1547/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4275 - binary_accuracy: 0.7500
    Epoch 1548/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4274 - binary_accuracy: 0.7500
    Epoch 1549/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4273 - binary_accuracy: 0.7500
    Epoch 1550/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4271 - binary_accuracy: 0.7500
    Epoch 1551/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4270 - binary_accuracy: 0.7500
    Epoch 1552/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4269 - binary_accuracy: 0.7500
    Epoch 1553/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4268 - binary_accuracy: 0.7500
    Epoch 1554/7000
    1/1 [==============================] - 0s 26ms/step - loss: 0.4267 - binary_accuracy: 0.7500
    Epoch 1555/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4266 - binary_accuracy: 0.7500
    Epoch 1556/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4265 - binary_accuracy: 0.7500
    Epoch 1557/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4264 - binary_accuracy: 0.7500
    Epoch 1558/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4263 - binary_accuracy: 0.7500
    Epoch 1559/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4262 - binary_accuracy: 0.7500
    Epoch 1560/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4261 - binary_accuracy: 0.7500
    Epoch 1561/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4260 - binary_accuracy: 0.7500
    Epoch 1562/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4258 - binary_accuracy: 0.7500
    Epoch 1563/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4257 - binary_accuracy: 0.7500
    Epoch 1564/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4256 - binary_accuracy: 0.7500
    Epoch 1565/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4255 - binary_accuracy: 0.7500
    Epoch 1566/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4254 - binary_accuracy: 0.7500
    Epoch 1567/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4253 - binary_accuracy: 0.7500
    Epoch 1568/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4252 - binary_accuracy: 0.7500
    Epoch 1569/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4251 - binary_accuracy: 0.7500
    Epoch 1570/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4250 - binary_accuracy: 0.7500
    Epoch 1571/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4249 - binary_accuracy: 0.7500
    Epoch 1572/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.4248 - binary_accuracy: 0.7500
    Epoch 1573/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4247 - binary_accuracy: 0.7500
    Epoch 1574/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4246 - binary_accuracy: 0.7500
    Epoch 1575/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4244 - binary_accuracy: 0.7500
    Epoch 1576/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4243 - binary_accuracy: 0.7500
    Epoch 1577/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4242 - binary_accuracy: 0.7500
    Epoch 1578/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4241 - binary_accuracy: 0.7500
    Epoch 1579/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4240 - binary_accuracy: 0.7500
    Epoch 1580/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4239 - binary_accuracy: 0.7500
    Epoch 1581/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4238 - binary_accuracy: 0.7500
    Epoch 1582/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4237 - binary_accuracy: 0.7500
    Epoch 1583/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4236 - binary_accuracy: 0.7500
    Epoch 1584/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4235 - binary_accuracy: 0.7500
    Epoch 1585/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4234 - binary_accuracy: 0.7500
    Epoch 1586/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4233 - binary_accuracy: 0.7500
    Epoch 1587/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4232 - binary_accuracy: 0.7500
    Epoch 1588/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4231 - binary_accuracy: 0.7500
    Epoch 1589/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4230 - binary_accuracy: 0.7500
    Epoch 1590/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4228 - binary_accuracy: 0.7500
    Epoch 1591/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4227 - binary_accuracy: 0.7500
    Epoch 1592/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4226 - binary_accuracy: 0.7500
    Epoch 1593/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4225 - binary_accuracy: 0.7500
    Epoch 1594/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4224 - binary_accuracy: 0.7500
    Epoch 1595/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4223 - binary_accuracy: 0.7500
    Epoch 1596/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4222 - binary_accuracy: 0.7500
    Epoch 1597/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4221 - binary_accuracy: 0.7500
    Epoch 1598/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4220 - binary_accuracy: 0.7500
    Epoch 1599/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4219 - binary_accuracy: 0.7500
    Epoch 1600/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4218 - binary_accuracy: 0.7500
    Epoch 1601/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.4217 - binary_accuracy: 0.7500
    Epoch 1602/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4216 - binary_accuracy: 0.7500
    Epoch 1603/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4215 - binary_accuracy: 0.7500
    Epoch 1604/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4214 - binary_accuracy: 0.7500
    Epoch 1605/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4213 - binary_accuracy: 0.7500
    Epoch 1606/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4212 - binary_accuracy: 0.7500
    Epoch 1607/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4210 - binary_accuracy: 0.7500
    Epoch 1608/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4209 - binary_accuracy: 0.7500
    Epoch 1609/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4208 - binary_accuracy: 0.7500
    Epoch 1610/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4207 - binary_accuracy: 0.7500
    Epoch 1611/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4206 - binary_accuracy: 0.7500
    Epoch 1612/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4205 - binary_accuracy: 0.7500
    Epoch 1613/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4204 - binary_accuracy: 0.7500
    Epoch 1614/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4203 - binary_accuracy: 0.7500
    Epoch 1615/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4202 - binary_accuracy: 0.7500
    Epoch 1616/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4201 - binary_accuracy: 0.7500
    Epoch 1617/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4200 - binary_accuracy: 0.7500
    Epoch 1618/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4199 - binary_accuracy: 0.7500
    Epoch 1619/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4198 - binary_accuracy: 0.7500
    Epoch 1620/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4197 - binary_accuracy: 0.7500
    Epoch 1621/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4196 - binary_accuracy: 0.7500
    Epoch 1622/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4195 - binary_accuracy: 0.7500
    Epoch 1623/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4194 - binary_accuracy: 0.7500
    Epoch 1624/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4193 - binary_accuracy: 0.7500
    Epoch 1625/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4192 - binary_accuracy: 0.7500
    Epoch 1626/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4191 - binary_accuracy: 0.7500
    Epoch 1627/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4189 - binary_accuracy: 0.7500
    Epoch 1628/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4188 - binary_accuracy: 0.7500
    Epoch 1629/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4187 - binary_accuracy: 0.7500
    Epoch 1630/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4186 - binary_accuracy: 0.7500
    Epoch 1631/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4185 - binary_accuracy: 0.7500
    Epoch 1632/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.4184 - binary_accuracy: 0.7500
    Epoch 1633/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4183 - binary_accuracy: 0.7500
    Epoch 1634/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4182 - binary_accuracy: 0.7500
    Epoch 1635/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4181 - binary_accuracy: 0.7500
    Epoch 1636/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4180 - binary_accuracy: 0.7500
    Epoch 1637/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4179 - binary_accuracy: 0.7500
    Epoch 1638/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4178 - binary_accuracy: 0.7500
    Epoch 1639/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4177 - binary_accuracy: 0.7500
    Epoch 1640/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4176 - binary_accuracy: 0.7500
    Epoch 1641/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4175 - binary_accuracy: 0.7500
    Epoch 1642/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4174 - binary_accuracy: 0.7500
    Epoch 1643/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.4173 - binary_accuracy: 0.7500
    Epoch 1644/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4172 - binary_accuracy: 0.7500
    Epoch 1645/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4171 - binary_accuracy: 0.7500
    Epoch 1646/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4170 - binary_accuracy: 0.7500
    Epoch 1647/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4169 - binary_accuracy: 0.7500
    Epoch 1648/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4168 - binary_accuracy: 0.7500
    Epoch 1649/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4167 - binary_accuracy: 0.7500
    Epoch 1650/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4166 - binary_accuracy: 0.7500
    Epoch 1651/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4165 - binary_accuracy: 0.7500
    Epoch 1652/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4164 - binary_accuracy: 0.7500
    Epoch 1653/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4163 - binary_accuracy: 0.7500
    Epoch 1654/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4162 - binary_accuracy: 0.7500
    Epoch 1655/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4161 - binary_accuracy: 0.7500
    Epoch 1656/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4160 - binary_accuracy: 0.7500
    Epoch 1657/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4158 - binary_accuracy: 0.7500
    Epoch 1658/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4157 - binary_accuracy: 0.7500
    Epoch 1659/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4156 - binary_accuracy: 0.7500
    Epoch 1660/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4155 - binary_accuracy: 0.7500
    Epoch 1661/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4154 - binary_accuracy: 0.7500
    Epoch 1662/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4153 - binary_accuracy: 0.7500
    Epoch 1663/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.4152 - binary_accuracy: 0.7500
    Epoch 1664/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4151 - binary_accuracy: 0.7500
    Epoch 1665/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4150 - binary_accuracy: 0.7500
    Epoch 1666/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4149 - binary_accuracy: 0.7500
    Epoch 1667/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4148 - binary_accuracy: 0.7500
    Epoch 1668/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4147 - binary_accuracy: 0.7500
    Epoch 1669/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4146 - binary_accuracy: 0.7500
    Epoch 1670/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4145 - binary_accuracy: 0.7500
    Epoch 1671/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4144 - binary_accuracy: 0.7500
    Epoch 1672/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4143 - binary_accuracy: 0.7500
    Epoch 1673/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4142 - binary_accuracy: 0.7500
    Epoch 1674/7000
    1/1 [==============================] - 0s 21ms/step - loss: 0.4141 - binary_accuracy: 0.7500
    Epoch 1675/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4140 - binary_accuracy: 0.7500
    Epoch 1676/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4139 - binary_accuracy: 0.7500
    Epoch 1677/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4138 - binary_accuracy: 0.7500
    Epoch 1678/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4137 - binary_accuracy: 0.7500
    Epoch 1679/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4136 - binary_accuracy: 0.7500
    Epoch 1680/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4135 - binary_accuracy: 0.7500
    Epoch 1681/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4134 - binary_accuracy: 0.7500
    Epoch 1682/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4133 - binary_accuracy: 0.7500
    Epoch 1683/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4132 - binary_accuracy: 0.7500
    Epoch 1684/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4131 - binary_accuracy: 0.7500
    Epoch 1685/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4130 - binary_accuracy: 0.7500
    Epoch 1686/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4129 - binary_accuracy: 0.7500
    Epoch 1687/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4128 - binary_accuracy: 0.7500
    Epoch 1688/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4127 - binary_accuracy: 0.7500
    Epoch 1689/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4126 - binary_accuracy: 0.7500
    Epoch 1690/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4125 - binary_accuracy: 0.7500
    Epoch 1691/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4124 - binary_accuracy: 0.7500
    Epoch 1692/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4123 - binary_accuracy: 0.7500
    Epoch 1693/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4122 - binary_accuracy: 0.7500
    Epoch 1694/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4121 - binary_accuracy: 0.7500
    Epoch 1695/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4120 - binary_accuracy: 0.7500
    Epoch 1696/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4119 - binary_accuracy: 0.7500
    Epoch 1697/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4118 - binary_accuracy: 0.7500
    Epoch 1698/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4117 - binary_accuracy: 0.7500
    Epoch 1699/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4116 - binary_accuracy: 0.7500
    Epoch 1700/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4115 - binary_accuracy: 0.7500
    Epoch 1701/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4114 - binary_accuracy: 0.7500
    Epoch 1702/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4113 - binary_accuracy: 0.7500
    Epoch 1703/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4112 - binary_accuracy: 0.7500
    Epoch 1704/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4111 - binary_accuracy: 0.7500
    Epoch 1705/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4110 - binary_accuracy: 0.7500
    Epoch 1706/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4109 - binary_accuracy: 0.7500
    Epoch 1707/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4108 - binary_accuracy: 0.7500
    Epoch 1708/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4107 - binary_accuracy: 0.7500
    Epoch 1709/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4106 - binary_accuracy: 0.7500
    Epoch 1710/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4105 - binary_accuracy: 0.7500
    Epoch 1711/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4104 - binary_accuracy: 0.7500
    Epoch 1712/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4103 - binary_accuracy: 0.7500
    Epoch 1713/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4102 - binary_accuracy: 0.7500
    Epoch 1714/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4101 - binary_accuracy: 0.7500
    Epoch 1715/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4100 - binary_accuracy: 0.7500
    Epoch 1716/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4099 - binary_accuracy: 0.7500
    Epoch 1717/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4098 - binary_accuracy: 0.7500
    Epoch 1718/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4097 - binary_accuracy: 0.7500
    Epoch 1719/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4096 - binary_accuracy: 0.7500
    Epoch 1720/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4095 - binary_accuracy: 0.7500
    Epoch 1721/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4094 - binary_accuracy: 0.7500
    Epoch 1722/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4093 - binary_accuracy: 0.7500
    Epoch 1723/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4092 - binary_accuracy: 0.7500
    Epoch 1724/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4091 - binary_accuracy: 0.7500
    Epoch 1725/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4090 - binary_accuracy: 0.7500
    Epoch 1726/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4089 - binary_accuracy: 0.7500
    Epoch 1727/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4088 - binary_accuracy: 0.7500
    Epoch 1728/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4087 - binary_accuracy: 0.7500
    Epoch 1729/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4086 - binary_accuracy: 0.7500
    Epoch 1730/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4085 - binary_accuracy: 0.7500
    Epoch 1731/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4084 - binary_accuracy: 0.7500
    Epoch 1732/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4083 - binary_accuracy: 0.7500
    Epoch 1733/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4082 - binary_accuracy: 0.7500
    Epoch 1734/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4081 - binary_accuracy: 0.7500
    Epoch 1735/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4080 - binary_accuracy: 0.7500
    Epoch 1736/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4079 - binary_accuracy: 0.7500
    Epoch 1737/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.4078 - binary_accuracy: 0.7500
    Epoch 1738/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4077 - binary_accuracy: 0.7500
    Epoch 1739/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4076 - binary_accuracy: 0.7500
    Epoch 1740/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4075 - binary_accuracy: 0.7500
    Epoch 1741/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4074 - binary_accuracy: 0.7500
    Epoch 1742/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4073 - binary_accuracy: 0.7500
    Epoch 1743/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4072 - binary_accuracy: 0.7500
    Epoch 1744/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4071 - binary_accuracy: 0.7500
    Epoch 1745/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4070 - binary_accuracy: 0.7500
    Epoch 1746/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.4069 - binary_accuracy: 0.7500
    Epoch 1747/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4068 - binary_accuracy: 0.7500
    Epoch 1748/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4067 - binary_accuracy: 0.7500
    Epoch 1749/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4067 - binary_accuracy: 0.7500
    Epoch 1750/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4066 - binary_accuracy: 0.7500
    Epoch 1751/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4065 - binary_accuracy: 0.7500
    Epoch 1752/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4064 - binary_accuracy: 0.7500
    Epoch 1753/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4063 - binary_accuracy: 0.7500
    Epoch 1754/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4062 - binary_accuracy: 0.7500
    Epoch 1755/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.4061 - binary_accuracy: 0.7500
    Epoch 1756/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4060 - binary_accuracy: 0.7500
    Epoch 1757/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4059 - binary_accuracy: 0.7500
    Epoch 1758/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4058 - binary_accuracy: 0.7500
    Epoch 1759/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4057 - binary_accuracy: 0.7500
    Epoch 1760/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4056 - binary_accuracy: 0.7500
    Epoch 1761/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4055 - binary_accuracy: 0.7500
    Epoch 1762/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4054 - binary_accuracy: 0.7500
    Epoch 1763/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4053 - binary_accuracy: 0.7500
    Epoch 1764/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4052 - binary_accuracy: 0.7500
    Epoch 1765/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.4051 - binary_accuracy: 0.7500
    Epoch 1766/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4050 - binary_accuracy: 0.7500
    Epoch 1767/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4049 - binary_accuracy: 0.7500
    Epoch 1768/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4048 - binary_accuracy: 0.7500
    Epoch 1769/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4047 - binary_accuracy: 0.7500
    Epoch 1770/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4046 - binary_accuracy: 0.7500
    Epoch 1771/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4045 - binary_accuracy: 0.7500
    Epoch 1772/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4044 - binary_accuracy: 0.7500
    Epoch 1773/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4043 - binary_accuracy: 0.7500
    Epoch 1774/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4042 - binary_accuracy: 0.7500
    Epoch 1775/7000
    1/1 [==============================] - 0s 41ms/step - loss: 0.4041 - binary_accuracy: 0.7500
    Epoch 1776/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4040 - binary_accuracy: 0.7500
    Epoch 1777/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4039 - binary_accuracy: 0.7500
    Epoch 1778/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4038 - binary_accuracy: 0.7500
    Epoch 1779/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4038 - binary_accuracy: 0.7500
    Epoch 1780/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4037 - binary_accuracy: 0.7500
    Epoch 1781/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4036 - binary_accuracy: 0.7500
    Epoch 1782/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4035 - binary_accuracy: 0.7500
    Epoch 1783/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4034 - binary_accuracy: 0.7500
    Epoch 1784/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4033 - binary_accuracy: 0.7500
    Epoch 1785/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4032 - binary_accuracy: 0.7500
    Epoch 1786/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4031 - binary_accuracy: 0.7500
    Epoch 1787/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4030 - binary_accuracy: 0.7500
    Epoch 1788/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4029 - binary_accuracy: 0.7500
    Epoch 1789/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4028 - binary_accuracy: 0.7500
    Epoch 1790/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4027 - binary_accuracy: 0.7500
    Epoch 1791/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4026 - binary_accuracy: 0.7500
    Epoch 1792/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4025 - binary_accuracy: 0.7500
    Epoch 1793/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4024 - binary_accuracy: 0.7500
    Epoch 1794/7000
    1/1 [==============================] - 0s 52ms/step - loss: 0.4023 - binary_accuracy: 0.7500
    Epoch 1795/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4022 - binary_accuracy: 0.7500
    Epoch 1796/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4021 - binary_accuracy: 0.7500
    Epoch 1797/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4020 - binary_accuracy: 0.7500
    Epoch 1798/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4019 - binary_accuracy: 0.7500
    Epoch 1799/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4018 - binary_accuracy: 0.7500
    Epoch 1800/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4017 - binary_accuracy: 0.7500
    Epoch 1801/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4017 - binary_accuracy: 0.7500
    Epoch 1802/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4016 - binary_accuracy: 0.7500
    Epoch 1803/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4015 - binary_accuracy: 0.7500
    Epoch 1804/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.4014 - binary_accuracy: 0.7500
    Epoch 1805/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.4013 - binary_accuracy: 0.7500
    Epoch 1806/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4012 - binary_accuracy: 0.7500
    Epoch 1807/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4011 - binary_accuracy: 0.7500
    Epoch 1808/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4010 - binary_accuracy: 0.7500
    Epoch 1809/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4009 - binary_accuracy: 0.7500
    Epoch 1810/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4008 - binary_accuracy: 0.7500
    Epoch 1811/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4007 - binary_accuracy: 0.7500
    Epoch 1812/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4006 - binary_accuracy: 0.7500
    Epoch 1813/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.4005 - binary_accuracy: 0.7500
    Epoch 1814/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.4004 - binary_accuracy: 0.7500
    Epoch 1815/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.4003 - binary_accuracy: 0.7500
    Epoch 1816/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4002 - binary_accuracy: 0.7500
    Epoch 1817/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4001 - binary_accuracy: 0.7500
    Epoch 1818/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.4000 - binary_accuracy: 0.7500
    Epoch 1819/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.4000 - binary_accuracy: 0.7500
    Epoch 1820/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3999 - binary_accuracy: 0.7500
    Epoch 1821/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3998 - binary_accuracy: 0.7500
    Epoch 1822/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3997 - binary_accuracy: 0.7500
    Epoch 1823/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3996 - binary_accuracy: 0.7500
    Epoch 1824/7000
    1/1 [==============================] - 0s 15ms/step - loss: 0.3995 - binary_accuracy: 0.7500
    Epoch 1825/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3994 - binary_accuracy: 0.7500
    Epoch 1826/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3993 - binary_accuracy: 0.7500
    Epoch 1827/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3992 - binary_accuracy: 0.7500
    Epoch 1828/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3991 - binary_accuracy: 0.7500
    Epoch 1829/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3990 - binary_accuracy: 0.7500
    Epoch 1830/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3989 - binary_accuracy: 0.7500
    Epoch 1831/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3988 - binary_accuracy: 0.7500
    Epoch 1832/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3987 - binary_accuracy: 0.7500
    Epoch 1833/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3986 - binary_accuracy: 0.7500
    Epoch 1834/7000
    1/1 [==============================] - 0s 31ms/step - loss: 0.3986 - binary_accuracy: 0.7500
    Epoch 1835/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3985 - binary_accuracy: 0.7500
    Epoch 1836/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3984 - binary_accuracy: 0.7500
    Epoch 1837/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3983 - binary_accuracy: 0.7500
    Epoch 1838/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3982 - binary_accuracy: 0.7500
    Epoch 1839/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3981 - binary_accuracy: 0.7500
    Epoch 1840/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3980 - binary_accuracy: 0.7500
    Epoch 1841/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3979 - binary_accuracy: 0.7500
    Epoch 1842/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3978 - binary_accuracy: 0.7500
    Epoch 1843/7000
    1/1 [==============================] - 0s 43ms/step - loss: 0.3977 - binary_accuracy: 0.7500
    Epoch 1844/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3976 - binary_accuracy: 0.7500
    Epoch 1845/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3975 - binary_accuracy: 0.7500
    Epoch 1846/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3974 - binary_accuracy: 0.7500
    Epoch 1847/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3973 - binary_accuracy: 0.7500
    Epoch 1848/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3973 - binary_accuracy: 0.7500
    Epoch 1849/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3972 - binary_accuracy: 0.7500
    Epoch 1850/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3971 - binary_accuracy: 0.7500
    Epoch 1851/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3970 - binary_accuracy: 0.7500
    Epoch 1852/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3969 - binary_accuracy: 0.7500
    Epoch 1853/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3968 - binary_accuracy: 0.7500
    Epoch 1854/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3967 - binary_accuracy: 0.7500
    Epoch 1855/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3966 - binary_accuracy: 0.7500
    Epoch 1856/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3965 - binary_accuracy: 0.7500
    Epoch 1857/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.3964 - binary_accuracy: 0.7500
    Epoch 1858/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3963 - binary_accuracy: 0.7500
    Epoch 1859/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3962 - binary_accuracy: 0.7500
    Epoch 1860/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3961 - binary_accuracy: 0.7500
    Epoch 1861/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3961 - binary_accuracy: 0.7500
    Epoch 1862/7000
    1/1 [==============================] - 0s 73ms/step - loss: 0.3960 - binary_accuracy: 0.7500
    Epoch 1863/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3959 - binary_accuracy: 0.7500
    Epoch 1864/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3958 - binary_accuracy: 0.7500
    Epoch 1865/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3957 - binary_accuracy: 0.7500
    Epoch 1866/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3956 - binary_accuracy: 0.7500
    Epoch 1867/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3955 - binary_accuracy: 0.7500
    Epoch 1868/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3954 - binary_accuracy: 0.7500
    Epoch 1869/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3953 - binary_accuracy: 0.7500
    Epoch 1870/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3952 - binary_accuracy: 0.7500
    Epoch 1871/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3951 - binary_accuracy: 0.7500
    Epoch 1872/7000
    1/1 [==============================] - 0s 32ms/step - loss: 0.3950 - binary_accuracy: 0.7500
    Epoch 1873/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3950 - binary_accuracy: 0.7500
    Epoch 1874/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3949 - binary_accuracy: 0.7500
    Epoch 1875/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3948 - binary_accuracy: 0.7500
    Epoch 1876/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3947 - binary_accuracy: 1.0000
    Epoch 1877/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3946 - binary_accuracy: 1.0000
    Epoch 1878/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3945 - binary_accuracy: 1.0000
    Epoch 1879/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3944 - binary_accuracy: 1.0000
    Epoch 1880/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3943 - binary_accuracy: 1.0000
    Epoch 1881/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3942 - binary_accuracy: 1.0000
    Epoch 1882/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3941 - binary_accuracy: 1.0000
    Epoch 1883/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3940 - binary_accuracy: 1.0000
    Epoch 1884/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3940 - binary_accuracy: 1.0000
    Epoch 1885/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3939 - binary_accuracy: 1.0000
    Epoch 1886/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3938 - binary_accuracy: 1.0000
    Epoch 1887/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3937 - binary_accuracy: 1.0000
    Epoch 1888/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3936 - binary_accuracy: 1.0000
    Epoch 1889/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3935 - binary_accuracy: 1.0000
    Epoch 1890/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3934 - binary_accuracy: 1.0000
    Epoch 1891/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3933 - binary_accuracy: 1.0000
    Epoch 1892/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3932 - binary_accuracy: 1.0000
    Epoch 1893/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3931 - binary_accuracy: 1.0000
    Epoch 1894/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3930 - binary_accuracy: 1.0000
    Epoch 1895/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3930 - binary_accuracy: 1.0000
    Epoch 1896/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3929 - binary_accuracy: 1.0000
    Epoch 1897/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3928 - binary_accuracy: 1.0000
    Epoch 1898/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3927 - binary_accuracy: 1.0000
    Epoch 1899/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3926 - binary_accuracy: 1.0000
    Epoch 1900/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3925 - binary_accuracy: 1.0000
    Epoch 1901/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3924 - binary_accuracy: 1.0000
    Epoch 1902/7000
    1/1 [==============================] - 0s 63ms/step - loss: 0.3923 - binary_accuracy: 1.0000
    Epoch 1903/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3922 - binary_accuracy: 1.0000
    Epoch 1904/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3921 - binary_accuracy: 1.0000
    Epoch 1905/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3921 - binary_accuracy: 1.0000
    Epoch 1906/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3920 - binary_accuracy: 1.0000
    Epoch 1907/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3919 - binary_accuracy: 1.0000
    Epoch 1908/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3918 - binary_accuracy: 1.0000
    Epoch 1909/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3917 - binary_accuracy: 1.0000
    Epoch 1910/7000
    1/1 [==============================] - 0s 16ms/step - loss: 0.3916 - binary_accuracy: 1.0000
    Epoch 1911/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.3915 - binary_accuracy: 1.0000
    Epoch 1912/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3914 - binary_accuracy: 1.0000
    Epoch 1913/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3913 - binary_accuracy: 1.0000
    Epoch 1914/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3912 - binary_accuracy: 1.0000
    Epoch 1915/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3912 - binary_accuracy: 1.0000
    Epoch 1916/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3911 - binary_accuracy: 1.0000
    Epoch 1917/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3910 - binary_accuracy: 1.0000
    Epoch 1918/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3909 - binary_accuracy: 1.0000
    Epoch 1919/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3908 - binary_accuracy: 1.0000
    Epoch 1920/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3907 - binary_accuracy: 1.0000
    Epoch 1921/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3906 - binary_accuracy: 1.0000
    Epoch 1922/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3905 - binary_accuracy: 1.0000
    Epoch 1923/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3904 - binary_accuracy: 1.0000
    Epoch 1924/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3904 - binary_accuracy: 1.0000
    Epoch 1925/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3903 - binary_accuracy: 1.0000
    Epoch 1926/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3902 - binary_accuracy: 1.0000
    Epoch 1927/7000
    1/1 [==============================] - 0s 67ms/step - loss: 0.3901 - binary_accuracy: 1.0000
    Epoch 1928/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3900 - binary_accuracy: 1.0000
    Epoch 1929/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3899 - binary_accuracy: 1.0000
    Epoch 1930/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3898 - binary_accuracy: 1.0000
    Epoch 1931/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3897 - binary_accuracy: 1.0000
    Epoch 1932/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3896 - binary_accuracy: 1.0000
    Epoch 1933/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3896 - binary_accuracy: 1.0000
    Epoch 1934/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3895 - binary_accuracy: 1.0000
    Epoch 1935/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3894 - binary_accuracy: 1.0000
    Epoch 1936/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.3893 - binary_accuracy: 1.0000
    Epoch 1937/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3892 - binary_accuracy: 1.0000
    Epoch 1938/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3891 - binary_accuracy: 1.0000
    Epoch 1939/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3890 - binary_accuracy: 1.0000
    Epoch 1940/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3889 - binary_accuracy: 1.0000
    Epoch 1941/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3889 - binary_accuracy: 1.0000
    Epoch 1942/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3888 - binary_accuracy: 1.0000
    Epoch 1943/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3887 - binary_accuracy: 1.0000
    Epoch 1944/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3886 - binary_accuracy: 1.0000
    Epoch 1945/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.3885 - binary_accuracy: 1.0000
    Epoch 1946/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3884 - binary_accuracy: 1.0000
    Epoch 1947/7000
    1/1 [==============================] - 0s 28ms/step - loss: 0.3883 - binary_accuracy: 1.0000
    Epoch 1948/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3882 - binary_accuracy: 1.0000
    Epoch 1949/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3881 - binary_accuracy: 1.0000
    Epoch 1950/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3881 - binary_accuracy: 1.0000
    Epoch 1951/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3880 - binary_accuracy: 1.0000
    Epoch 1952/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3879 - binary_accuracy: 1.0000
    Epoch 1953/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3878 - binary_accuracy: 1.0000
    Epoch 1954/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3877 - binary_accuracy: 1.0000
    Epoch 1955/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3876 - binary_accuracy: 1.0000
    Epoch 1956/7000
    1/1 [==============================] - 0s 58ms/step - loss: 0.3875 - binary_accuracy: 1.0000
    Epoch 1957/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3874 - binary_accuracy: 1.0000
    Epoch 1958/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3874 - binary_accuracy: 1.0000
    Epoch 1959/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3873 - binary_accuracy: 1.0000
    Epoch 1960/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3872 - binary_accuracy: 1.0000
    Epoch 1961/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3871 - binary_accuracy: 1.0000
    Epoch 1962/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3870 - binary_accuracy: 1.0000
    Epoch 1963/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3869 - binary_accuracy: 1.0000
    Epoch 1964/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3868 - binary_accuracy: 1.0000
    Epoch 1965/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3867 - binary_accuracy: 1.0000
    Epoch 1966/7000
    1/1 [==============================] - 0s 26ms/step - loss: 0.3867 - binary_accuracy: 1.0000
    Epoch 1967/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3866 - binary_accuracy: 1.0000
    Epoch 1968/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3865 - binary_accuracy: 1.0000
    Epoch 1969/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3864 - binary_accuracy: 1.0000
    Epoch 1970/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3863 - binary_accuracy: 1.0000
    Epoch 1971/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3862 - binary_accuracy: 1.0000
    Epoch 1972/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3861 - binary_accuracy: 1.0000
    Epoch 1973/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3861 - binary_accuracy: 1.0000
    Epoch 1974/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3860 - binary_accuracy: 1.0000
    Epoch 1975/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.3859 - binary_accuracy: 1.0000
    Epoch 1976/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3858 - binary_accuracy: 1.0000
    Epoch 1977/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3857 - binary_accuracy: 1.0000
    Epoch 1978/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3856 - binary_accuracy: 1.0000
    Epoch 1979/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3855 - binary_accuracy: 1.0000
    Epoch 1980/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3854 - binary_accuracy: 1.0000
    Epoch 1981/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3854 - binary_accuracy: 1.0000
    Epoch 1982/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3853 - binary_accuracy: 1.0000
    Epoch 1983/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3852 - binary_accuracy: 1.0000
    Epoch 1984/7000
    1/1 [==============================] - 0s 51ms/step - loss: 0.3851 - binary_accuracy: 1.0000
    Epoch 1985/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3850 - binary_accuracy: 1.0000
    Epoch 1986/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3849 - binary_accuracy: 1.0000
    Epoch 1987/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3848 - binary_accuracy: 1.0000
    Epoch 1988/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3848 - binary_accuracy: 1.0000
    Epoch 1989/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3847 - binary_accuracy: 1.0000
    Epoch 1990/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3846 - binary_accuracy: 1.0000
    Epoch 1991/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3845 - binary_accuracy: 1.0000
    Epoch 1992/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3844 - binary_accuracy: 1.0000
    Epoch 1993/7000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3843 - binary_accuracy: 1.0000
    Epoch 1994/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3842 - binary_accuracy: 1.0000
    Epoch 1995/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3842 - binary_accuracy: 1.0000
    Epoch 1996/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3841 - binary_accuracy: 1.0000
    Epoch 1997/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3840 - binary_accuracy: 1.0000
    Epoch 1998/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3839 - binary_accuracy: 1.0000
    Epoch 1999/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3838 - binary_accuracy: 1.0000
    Epoch 2000/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3837 - binary_accuracy: 1.0000
    Epoch 2001/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3836 - binary_accuracy: 1.0000
    Epoch 2002/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3836 - binary_accuracy: 1.0000
    Epoch 2003/7000
    1/1 [==============================] - 0s 36ms/step - loss: 0.3835 - binary_accuracy: 1.0000
    Epoch 2004/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3834 - binary_accuracy: 1.0000
    Epoch 2005/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3833 - binary_accuracy: 1.0000
    Epoch 2006/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3832 - binary_accuracy: 1.0000
    Epoch 2007/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3831 - binary_accuracy: 1.0000
    Epoch 2008/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3830 - binary_accuracy: 1.0000
    Epoch 2009/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3830 - binary_accuracy: 1.0000
    Epoch 2010/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3829 - binary_accuracy: 1.0000
    Epoch 2011/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3828 - binary_accuracy: 1.0000
    Epoch 2012/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3827 - binary_accuracy: 1.0000
    Epoch 2013/7000
    1/1 [==============================] - 0s 18ms/step - loss: 0.3826 - binary_accuracy: 1.0000
    Epoch 2014/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3825 - binary_accuracy: 1.0000
    Epoch 2015/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3824 - binary_accuracy: 1.0000
    Epoch 2016/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3824 - binary_accuracy: 1.0000
    Epoch 2017/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3823 - binary_accuracy: 1.0000
    Epoch 2018/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3822 - binary_accuracy: 1.0000
    Epoch 2019/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3821 - binary_accuracy: 1.0000
    Epoch 2020/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3820 - binary_accuracy: 1.0000
    Epoch 2021/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3819 - binary_accuracy: 1.0000
    Epoch 2022/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3819 - binary_accuracy: 1.0000
    Epoch 2023/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.3818 - binary_accuracy: 1.0000
    Epoch 2024/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3817 - binary_accuracy: 1.0000
    Epoch 2025/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3816 - binary_accuracy: 1.0000
    Epoch 2026/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3815 - binary_accuracy: 1.0000
    Epoch 2027/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3814 - binary_accuracy: 1.0000
    Epoch 2028/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3813 - binary_accuracy: 1.0000
    Epoch 2029/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3813 - binary_accuracy: 1.0000
    Epoch 2030/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3812 - binary_accuracy: 1.0000
    Epoch 2031/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3811 - binary_accuracy: 1.0000
    Epoch 2032/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3810 - binary_accuracy: 1.0000
    Epoch 2033/7000
    1/1 [==============================] - 0s 33ms/step - loss: 0.3809 - binary_accuracy: 1.0000
    Epoch 2034/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3808 - binary_accuracy: 1.0000
    Epoch 2035/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3808 - binary_accuracy: 1.0000
    Epoch 2036/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3807 - binary_accuracy: 1.0000
    Epoch 2037/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3806 - binary_accuracy: 1.0000
    Epoch 2038/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3805 - binary_accuracy: 1.0000
    Epoch 2039/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3804 - binary_accuracy: 1.0000
    Epoch 2040/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3803 - binary_accuracy: 1.0000
    Epoch 2041/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3803 - binary_accuracy: 1.0000
    Epoch 2042/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3802 - binary_accuracy: 1.0000
    Epoch 2043/7000
    1/1 [==============================] - 0s 78ms/step - loss: 0.3801 - binary_accuracy: 1.0000
    Epoch 2044/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3800 - binary_accuracy: 1.0000
    Epoch 2045/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3799 - binary_accuracy: 1.0000
    Epoch 2046/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3798 - binary_accuracy: 1.0000
    Epoch 2047/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3798 - binary_accuracy: 1.0000
    Epoch 2048/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3797 - binary_accuracy: 1.0000
    Epoch 2049/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3796 - binary_accuracy: 1.0000
    Epoch 2050/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3795 - binary_accuracy: 1.0000
    Epoch 2051/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.3794 - binary_accuracy: 1.0000
    Epoch 2052/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3793 - binary_accuracy: 1.0000
    Epoch 2053/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3793 - binary_accuracy: 1.0000
    Epoch 2054/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3792 - binary_accuracy: 1.0000
    Epoch 2055/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3791 - binary_accuracy: 1.0000
    Epoch 2056/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3790 - binary_accuracy: 1.0000
    Epoch 2057/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3789 - binary_accuracy: 1.0000
    Epoch 2058/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3788 - binary_accuracy: 1.0000
    Epoch 2059/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3788 - binary_accuracy: 1.0000
    Epoch 2060/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3787 - binary_accuracy: 1.0000
    Epoch 2061/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3786 - binary_accuracy: 1.0000
    Epoch 2062/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3785 - binary_accuracy: 1.0000
    Epoch 2063/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3784 - binary_accuracy: 1.0000
    Epoch 2064/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3783 - binary_accuracy: 1.0000
    Epoch 2065/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3783 - binary_accuracy: 1.0000
    Epoch 2066/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3782 - binary_accuracy: 1.0000
    Epoch 2067/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3781 - binary_accuracy: 1.0000
    Epoch 2068/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3780 - binary_accuracy: 1.0000
    Epoch 2069/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3779 - binary_accuracy: 1.0000
    Epoch 2070/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3778 - binary_accuracy: 1.0000
    Epoch 2071/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3778 - binary_accuracy: 1.0000
    Epoch 2072/7000
    1/1 [==============================] - 0s 16ms/step - loss: 0.3777 - binary_accuracy: 1.0000
    Epoch 2073/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3776 - binary_accuracy: 1.0000
    Epoch 2074/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3775 - binary_accuracy: 1.0000
    Epoch 2075/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3774 - binary_accuracy: 1.0000
    Epoch 2076/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3773 - binary_accuracy: 1.0000
    Epoch 2077/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3773 - binary_accuracy: 1.0000
    Epoch 2078/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3772 - binary_accuracy: 1.0000
    Epoch 2079/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3771 - binary_accuracy: 1.0000
    Epoch 2080/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3770 - binary_accuracy: 1.0000
    Epoch 2081/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3769 - binary_accuracy: 1.0000
    Epoch 2082/7000
    1/1 [==============================] - 0s 20ms/step - loss: 0.3769 - binary_accuracy: 1.0000
    Epoch 2083/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3768 - binary_accuracy: 1.0000
    Epoch 2084/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3767 - binary_accuracy: 1.0000
    Epoch 2085/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3766 - binary_accuracy: 1.0000
    Epoch 2086/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3765 - binary_accuracy: 1.0000
    Epoch 2087/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3764 - binary_accuracy: 1.0000
    Epoch 2088/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3764 - binary_accuracy: 1.0000
    Epoch 2089/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3763 - binary_accuracy: 1.0000
    Epoch 2090/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3762 - binary_accuracy: 1.0000
    Epoch 2091/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.3761 - binary_accuracy: 1.0000
    Epoch 2092/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3760 - binary_accuracy: 1.0000
    Epoch 2093/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3760 - binary_accuracy: 1.0000
    Epoch 2094/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3759 - binary_accuracy: 1.0000
    Epoch 2095/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3758 - binary_accuracy: 1.0000
    Epoch 2096/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3757 - binary_accuracy: 1.0000
    Epoch 2097/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3756 - binary_accuracy: 1.0000
    Epoch 2098/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3755 - binary_accuracy: 1.0000
    Epoch 2099/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3755 - binary_accuracy: 1.0000
    Epoch 2100/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3754 - binary_accuracy: 1.0000
    Epoch 2101/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.3753 - binary_accuracy: 1.0000
    Epoch 2102/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3752 - binary_accuracy: 1.0000
    Epoch 2103/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3751 - binary_accuracy: 1.0000
    Epoch 2104/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3751 - binary_accuracy: 1.0000
    Epoch 2105/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.3750 - binary_accuracy: 1.0000
    Epoch 2106/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3749 - binary_accuracy: 1.0000
    Epoch 2107/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3748 - binary_accuracy: 1.0000
    Epoch 2108/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3747 - binary_accuracy: 1.0000
    Epoch 2109/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3747 - binary_accuracy: 1.0000
    Epoch 2110/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3746 - binary_accuracy: 1.0000
    Epoch 2111/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3745 - binary_accuracy: 1.0000
    Epoch 2112/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3744 - binary_accuracy: 1.0000
    Epoch 2113/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3743 - binary_accuracy: 1.0000
    Epoch 2114/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3742 - binary_accuracy: 1.0000
    Epoch 2115/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3742 - binary_accuracy: 1.0000
    Epoch 2116/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3741 - binary_accuracy: 1.0000
    Epoch 2117/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3740 - binary_accuracy: 1.0000
    Epoch 2118/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3739 - binary_accuracy: 1.0000
    Epoch 2119/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3738 - binary_accuracy: 1.0000
    Epoch 2120/7000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3738 - binary_accuracy: 1.0000
    Epoch 2121/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3737 - binary_accuracy: 1.0000
    Epoch 2122/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3736 - binary_accuracy: 1.0000
    Epoch 2123/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3735 - binary_accuracy: 1.0000
    Epoch 2124/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3734 - binary_accuracy: 1.0000
    Epoch 2125/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3734 - binary_accuracy: 1.0000
    Epoch 2126/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3733 - binary_accuracy: 1.0000
    Epoch 2127/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3732 - binary_accuracy: 1.0000
    Epoch 2128/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3731 - binary_accuracy: 1.0000
    Epoch 2129/7000
    1/1 [==============================] - 0s 29ms/step - loss: 0.3730 - binary_accuracy: 1.0000
    Epoch 2130/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3730 - binary_accuracy: 1.0000
    Epoch 2131/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3729 - binary_accuracy: 1.0000
    Epoch 2132/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3728 - binary_accuracy: 1.0000
    Epoch 2133/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3727 - binary_accuracy: 1.0000
    Epoch 2134/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3726 - binary_accuracy: 1.0000
    Epoch 2135/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3726 - binary_accuracy: 1.0000
    Epoch 2136/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3725 - binary_accuracy: 1.0000
    Epoch 2137/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3724 - binary_accuracy: 1.0000
    Epoch 2138/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.3723 - binary_accuracy: 1.0000
    Epoch 2139/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3722 - binary_accuracy: 1.0000
    Epoch 2140/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3722 - binary_accuracy: 1.0000
    Epoch 2141/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3721 - binary_accuracy: 1.0000
    Epoch 2142/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3720 - binary_accuracy: 1.0000
    Epoch 2143/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3719 - binary_accuracy: 1.0000
    Epoch 2144/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3718 - binary_accuracy: 1.0000
    Epoch 2145/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3718 - binary_accuracy: 1.0000
    Epoch 2146/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3717 - binary_accuracy: 1.0000
    Epoch 2147/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3716 - binary_accuracy: 1.0000
    Epoch 2148/7000
    1/1 [==============================] - 0s 22ms/step - loss: 0.3715 - binary_accuracy: 1.0000
    Epoch 2149/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3714 - binary_accuracy: 1.0000
    Epoch 2150/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3714 - binary_accuracy: 1.0000
    Epoch 2151/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3713 - binary_accuracy: 1.0000
    Epoch 2152/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3712 - binary_accuracy: 1.0000
    Epoch 2153/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3711 - binary_accuracy: 1.0000
    Epoch 2154/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3710 - binary_accuracy: 1.0000
    Epoch 2155/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3710 - binary_accuracy: 1.0000
    Epoch 2156/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3709 - binary_accuracy: 1.0000
    Epoch 2157/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3708 - binary_accuracy: 1.0000
    Epoch 2158/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3707 - binary_accuracy: 1.0000
    Epoch 2159/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3706 - binary_accuracy: 1.0000
    Epoch 2160/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3706 - binary_accuracy: 1.0000
    Epoch 2161/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3705 - binary_accuracy: 1.0000
    Epoch 2162/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3704 - binary_accuracy: 1.0000
    Epoch 2163/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3703 - binary_accuracy: 1.0000
    Epoch 2164/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3703 - binary_accuracy: 1.0000
    Epoch 2165/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3702 - binary_accuracy: 1.0000
    Epoch 2166/7000
    1/1 [==============================] - 0s 16ms/step - loss: 0.3701 - binary_accuracy: 1.0000
    Epoch 2167/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3700 - binary_accuracy: 1.0000
    Epoch 2168/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3699 - binary_accuracy: 1.0000
    Epoch 2169/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3699 - binary_accuracy: 1.0000
    Epoch 2170/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3698 - binary_accuracy: 1.0000
    Epoch 2171/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3697 - binary_accuracy: 1.0000
    Epoch 2172/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3696 - binary_accuracy: 1.0000
    Epoch 2173/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3695 - binary_accuracy: 1.0000
    Epoch 2174/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3695 - binary_accuracy: 1.0000
    Epoch 2175/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3694 - binary_accuracy: 1.0000
    Epoch 2176/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3693 - binary_accuracy: 1.0000
    Epoch 2177/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3692 - binary_accuracy: 1.0000
    Epoch 2178/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3691 - binary_accuracy: 1.0000
    Epoch 2179/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3691 - binary_accuracy: 1.0000
    Epoch 2180/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3690 - binary_accuracy: 1.0000
    Epoch 2181/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3689 - binary_accuracy: 1.0000
    Epoch 2182/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3688 - binary_accuracy: 1.0000
    Epoch 2183/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.3688 - binary_accuracy: 1.0000
    Epoch 2184/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3687 - binary_accuracy: 1.0000
    Epoch 2185/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3686 - binary_accuracy: 1.0000
    Epoch 2186/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3685 - binary_accuracy: 1.0000
    Epoch 2187/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3684 - binary_accuracy: 1.0000
    Epoch 2188/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3684 - binary_accuracy: 1.0000
    Epoch 2189/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3683 - binary_accuracy: 1.0000
    Epoch 2190/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3682 - binary_accuracy: 1.0000
    Epoch 2191/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.3681 - binary_accuracy: 1.0000
    Epoch 2192/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3681 - binary_accuracy: 1.0000
    Epoch 2193/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3680 - binary_accuracy: 1.0000
    Epoch 2194/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3679 - binary_accuracy: 1.0000
    Epoch 2195/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3678 - binary_accuracy: 1.0000
    Epoch 2196/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3677 - binary_accuracy: 1.0000
    Epoch 2197/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3677 - binary_accuracy: 1.0000
    Epoch 2198/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3676 - binary_accuracy: 1.0000
    Epoch 2199/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3675 - binary_accuracy: 1.0000
    Epoch 2200/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3674 - binary_accuracy: 1.0000
    Epoch 2201/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3674 - binary_accuracy: 1.0000
    Epoch 2202/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3673 - binary_accuracy: 1.0000
    Epoch 2203/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3672 - binary_accuracy: 1.0000
    Epoch 2204/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3671 - binary_accuracy: 1.0000
    Epoch 2205/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3670 - binary_accuracy: 1.0000
    Epoch 2206/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3670 - binary_accuracy: 1.0000
    Epoch 2207/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3669 - binary_accuracy: 1.0000
    Epoch 2208/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3668 - binary_accuracy: 1.0000
    Epoch 2209/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3667 - binary_accuracy: 1.0000
    Epoch 2210/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3667 - binary_accuracy: 1.0000
    Epoch 2211/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3666 - binary_accuracy: 1.0000
    Epoch 2212/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3665 - binary_accuracy: 1.0000
    Epoch 2213/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3664 - binary_accuracy: 1.0000
    Epoch 2214/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3664 - binary_accuracy: 1.0000
    Epoch 2215/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3663 - binary_accuracy: 1.0000
    Epoch 2216/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3662 - binary_accuracy: 1.0000
    Epoch 2217/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3661 - binary_accuracy: 1.0000
    Epoch 2218/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3660 - binary_accuracy: 1.0000
    Epoch 2219/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3660 - binary_accuracy: 1.0000
    Epoch 2220/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3659 - binary_accuracy: 1.0000
    Epoch 2221/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3658 - binary_accuracy: 1.0000
    Epoch 2222/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3657 - binary_accuracy: 1.0000
    Epoch 2223/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3657 - binary_accuracy: 1.0000
    Epoch 2224/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3656 - binary_accuracy: 1.0000
    Epoch 2225/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3655 - binary_accuracy: 1.0000
    Epoch 2226/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3654 - binary_accuracy: 1.0000
    Epoch 2227/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3654 - binary_accuracy: 1.0000
    Epoch 2228/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3653 - binary_accuracy: 1.0000
    Epoch 2229/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3652 - binary_accuracy: 1.0000
    Epoch 2230/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3651 - binary_accuracy: 1.0000
    Epoch 2231/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3650 - binary_accuracy: 1.0000
    Epoch 2232/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3650 - binary_accuracy: 1.0000
    Epoch 2233/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3649 - binary_accuracy: 1.0000
    Epoch 2234/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.3648 - binary_accuracy: 1.0000
    Epoch 2235/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3647 - binary_accuracy: 1.0000
    Epoch 2236/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3647 - binary_accuracy: 1.0000
    Epoch 2237/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3646 - binary_accuracy: 1.0000
    Epoch 2238/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3645 - binary_accuracy: 1.0000
    Epoch 2239/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3644 - binary_accuracy: 1.0000
    Epoch 2240/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3644 - binary_accuracy: 1.0000
    Epoch 2241/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3643 - binary_accuracy: 1.0000
    Epoch 2242/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3642 - binary_accuracy: 1.0000
    Epoch 2243/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3641 - binary_accuracy: 1.0000
    Epoch 2244/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3641 - binary_accuracy: 1.0000
    Epoch 2245/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3640 - binary_accuracy: 1.0000
    Epoch 2246/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3639 - binary_accuracy: 1.0000
    Epoch 2247/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3638 - binary_accuracy: 1.0000
    Epoch 2248/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3638 - binary_accuracy: 1.0000
    Epoch 2249/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3637 - binary_accuracy: 1.0000
    Epoch 2250/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3636 - binary_accuracy: 1.0000
    Epoch 2251/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3635 - binary_accuracy: 1.0000
    Epoch 2252/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3634 - binary_accuracy: 1.0000
    Epoch 2253/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3634 - binary_accuracy: 1.0000
    Epoch 2254/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3633 - binary_accuracy: 1.0000
    Epoch 2255/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3632 - binary_accuracy: 1.0000
    Epoch 2256/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3631 - binary_accuracy: 1.0000
    Epoch 2257/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3631 - binary_accuracy: 1.0000
    Epoch 2258/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3630 - binary_accuracy: 1.0000
    Epoch 2259/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3629 - binary_accuracy: 1.0000
    Epoch 2260/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3628 - binary_accuracy: 1.0000
    Epoch 2261/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3628 - binary_accuracy: 1.0000
    Epoch 2262/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3627 - binary_accuracy: 1.0000
    Epoch 2263/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3626 - binary_accuracy: 1.0000
    Epoch 2264/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3625 - binary_accuracy: 1.0000
    Epoch 2265/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3625 - binary_accuracy: 1.0000
    Epoch 2266/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3624 - binary_accuracy: 1.0000
    Epoch 2267/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3623 - binary_accuracy: 1.0000
    Epoch 2268/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3622 - binary_accuracy: 1.0000
    Epoch 2269/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3622 - binary_accuracy: 1.0000
    Epoch 2270/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3621 - binary_accuracy: 1.0000
    Epoch 2271/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.3620 - binary_accuracy: 1.0000
    Epoch 2272/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3619 - binary_accuracy: 1.0000
    Epoch 2273/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3619 - binary_accuracy: 1.0000
    Epoch 2274/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3618 - binary_accuracy: 1.0000
    Epoch 2275/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3617 - binary_accuracy: 1.0000
    Epoch 2276/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3616 - binary_accuracy: 1.0000
    Epoch 2277/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3616 - binary_accuracy: 1.0000
    Epoch 2278/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3615 - binary_accuracy: 1.0000
    Epoch 2279/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3614 - binary_accuracy: 1.0000
    Epoch 2280/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3613 - binary_accuracy: 1.0000
    Epoch 2281/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3613 - binary_accuracy: 1.0000
    Epoch 2282/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3612 - binary_accuracy: 1.0000
    Epoch 2283/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3611 - binary_accuracy: 1.0000
    Epoch 2284/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3610 - binary_accuracy: 1.0000
    Epoch 2285/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3610 - binary_accuracy: 1.0000
    Epoch 2286/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3609 - binary_accuracy: 1.0000
    Epoch 2287/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3608 - binary_accuracy: 1.0000
    Epoch 2288/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3607 - binary_accuracy: 1.0000
    Epoch 2289/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3607 - binary_accuracy: 1.0000
    Epoch 2290/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3606 - binary_accuracy: 1.0000
    Epoch 2291/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3605 - binary_accuracy: 1.0000
    Epoch 2292/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3604 - binary_accuracy: 1.0000
    Epoch 2293/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3604 - binary_accuracy: 1.0000
    Epoch 2294/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3603 - binary_accuracy: 1.0000
    Epoch 2295/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3602 - binary_accuracy: 1.0000
    Epoch 2296/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3602 - binary_accuracy: 1.0000
    Epoch 2297/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3601 - binary_accuracy: 1.0000
    Epoch 2298/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3600 - binary_accuracy: 1.0000
    Epoch 2299/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3599 - binary_accuracy: 1.0000
    Epoch 2300/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.3599 - binary_accuracy: 1.0000
    Epoch 2301/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3598 - binary_accuracy: 1.0000
    Epoch 2302/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3597 - binary_accuracy: 1.0000
    Epoch 2303/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3596 - binary_accuracy: 1.0000
    Epoch 2304/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3596 - binary_accuracy: 1.0000
    Epoch 2305/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3595 - binary_accuracy: 1.0000
    Epoch 2306/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3594 - binary_accuracy: 1.0000
    Epoch 2307/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3593 - binary_accuracy: 1.0000
    Epoch 2308/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3593 - binary_accuracy: 1.0000
    Epoch 2309/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3592 - binary_accuracy: 1.0000
    Epoch 2310/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3591 - binary_accuracy: 1.0000
    Epoch 2311/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3590 - binary_accuracy: 1.0000
    Epoch 2312/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3590 - binary_accuracy: 1.0000
    Epoch 2313/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3589 - binary_accuracy: 1.0000
    Epoch 2314/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3588 - binary_accuracy: 1.0000
    Epoch 2315/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3587 - binary_accuracy: 1.0000
    Epoch 2316/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3587 - binary_accuracy: 1.0000
    Epoch 2317/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3586 - binary_accuracy: 1.0000
    Epoch 2318/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3585 - binary_accuracy: 1.0000
    Epoch 2319/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3585 - binary_accuracy: 1.0000
    Epoch 2320/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3584 - binary_accuracy: 1.0000
    Epoch 2321/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3583 - binary_accuracy: 1.0000
    Epoch 2322/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3582 - binary_accuracy: 1.0000
    Epoch 2323/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3582 - binary_accuracy: 1.0000
    Epoch 2324/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3581 - binary_accuracy: 1.0000
    Epoch 2325/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3580 - binary_accuracy: 1.0000
    Epoch 2326/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3579 - binary_accuracy: 1.0000
    Epoch 2327/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3579 - binary_accuracy: 1.0000
    Epoch 2328/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3578 - binary_accuracy: 1.0000
    Epoch 2329/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3577 - binary_accuracy: 1.0000
    Epoch 2330/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3576 - binary_accuracy: 1.0000
    Epoch 2331/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3576 - binary_accuracy: 1.0000
    Epoch 2332/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3575 - binary_accuracy: 1.0000
    Epoch 2333/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3574 - binary_accuracy: 1.0000
    Epoch 2334/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3574 - binary_accuracy: 1.0000
    Epoch 2335/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3573 - binary_accuracy: 1.0000
    Epoch 2336/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.3572 - binary_accuracy: 1.0000
    Epoch 2337/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3571 - binary_accuracy: 1.0000
    Epoch 2338/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3571 - binary_accuracy: 1.0000
    Epoch 2339/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3570 - binary_accuracy: 1.0000
    Epoch 2340/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3569 - binary_accuracy: 1.0000
    Epoch 2341/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3568 - binary_accuracy: 1.0000
    Epoch 2342/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3568 - binary_accuracy: 1.0000
    Epoch 2343/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3567 - binary_accuracy: 1.0000
    Epoch 2344/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3566 - binary_accuracy: 1.0000
    Epoch 2345/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3566 - binary_accuracy: 1.0000
    Epoch 2346/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3565 - binary_accuracy: 1.0000
    Epoch 2347/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3564 - binary_accuracy: 1.0000
    Epoch 2348/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3563 - binary_accuracy: 1.0000
    Epoch 2349/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3563 - binary_accuracy: 1.0000
    Epoch 2350/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3562 - binary_accuracy: 1.0000
    Epoch 2351/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3561 - binary_accuracy: 1.0000
    Epoch 2352/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3561 - binary_accuracy: 1.0000
    Epoch 2353/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3560 - binary_accuracy: 1.0000
    Epoch 2354/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3559 - binary_accuracy: 1.0000
    Epoch 2355/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3558 - binary_accuracy: 1.0000
    Epoch 2356/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3558 - binary_accuracy: 1.0000
    Epoch 2357/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3557 - binary_accuracy: 1.0000
    Epoch 2358/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3556 - binary_accuracy: 1.0000
    Epoch 2359/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3555 - binary_accuracy: 1.0000
    Epoch 2360/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3555 - binary_accuracy: 1.0000
    Epoch 2361/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3554 - binary_accuracy: 1.0000
    Epoch 2362/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3553 - binary_accuracy: 1.0000
    Epoch 2363/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3553 - binary_accuracy: 1.0000
    Epoch 2364/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3552 - binary_accuracy: 1.0000
    Epoch 2365/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3551 - binary_accuracy: 1.0000
    Epoch 2366/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3550 - binary_accuracy: 1.0000
    Epoch 2367/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3550 - binary_accuracy: 1.0000
    Epoch 2368/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3549 - binary_accuracy: 1.0000
    Epoch 2369/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3548 - binary_accuracy: 1.0000
    Epoch 2370/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3548 - binary_accuracy: 1.0000
    Epoch 2371/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3547 - binary_accuracy: 1.0000
    Epoch 2372/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3546 - binary_accuracy: 1.0000
    Epoch 2373/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3545 - binary_accuracy: 1.0000
    Epoch 2374/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3545 - binary_accuracy: 1.0000
    Epoch 2375/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3544 - binary_accuracy: 1.0000
    Epoch 2376/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3543 - binary_accuracy: 1.0000
    Epoch 2377/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3543 - binary_accuracy: 1.0000
    Epoch 2378/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3542 - binary_accuracy: 1.0000
    Epoch 2379/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3541 - binary_accuracy: 1.0000
    Epoch 2380/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3540 - binary_accuracy: 1.0000
    Epoch 2381/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3540 - binary_accuracy: 1.0000
    Epoch 2382/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3539 - binary_accuracy: 1.0000
    Epoch 2383/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.3538 - binary_accuracy: 1.0000
    Epoch 2384/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3538 - binary_accuracy: 1.0000
    Epoch 2385/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3537 - binary_accuracy: 1.0000
    Epoch 2386/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3536 - binary_accuracy: 1.0000
    Epoch 2387/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3535 - binary_accuracy: 1.0000
    Epoch 2388/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3535 - binary_accuracy: 1.0000
    Epoch 2389/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3534 - binary_accuracy: 1.0000
    Epoch 2390/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3533 - binary_accuracy: 1.0000
    Epoch 2391/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3533 - binary_accuracy: 1.0000
    Epoch 2392/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3532 - binary_accuracy: 1.0000
    Epoch 2393/7000
    1/1 [==============================] - 0s 18ms/step - loss: 0.3531 - binary_accuracy: 1.0000
    Epoch 2394/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3530 - binary_accuracy: 1.0000
    Epoch 2395/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3530 - binary_accuracy: 1.0000
    Epoch 2396/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3529 - binary_accuracy: 1.0000
    Epoch 2397/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3528 - binary_accuracy: 1.0000
    Epoch 2398/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3528 - binary_accuracy: 1.0000
    Epoch 2399/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3527 - binary_accuracy: 1.0000
    Epoch 2400/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3526 - binary_accuracy: 1.0000
    Epoch 2401/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3525 - binary_accuracy: 1.0000
    Epoch 2402/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3525 - binary_accuracy: 1.0000
    Epoch 2403/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.3524 - binary_accuracy: 1.0000
    Epoch 2404/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3523 - binary_accuracy: 1.0000
    Epoch 2405/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3523 - binary_accuracy: 1.0000
    Epoch 2406/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3522 - binary_accuracy: 1.0000
    Epoch 2407/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3521 - binary_accuracy: 1.0000
    Epoch 2408/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3521 - binary_accuracy: 1.0000
    Epoch 2409/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3520 - binary_accuracy: 1.0000
    Epoch 2410/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3519 - binary_accuracy: 1.0000
    Epoch 2411/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3518 - binary_accuracy: 1.0000
    Epoch 2412/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3518 - binary_accuracy: 1.0000
    Epoch 2413/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3517 - binary_accuracy: 1.0000
    Epoch 2414/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3516 - binary_accuracy: 1.0000
    Epoch 2415/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3516 - binary_accuracy: 1.0000
    Epoch 2416/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3515 - binary_accuracy: 1.0000
    Epoch 2417/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3514 - binary_accuracy: 1.0000
    Epoch 2418/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3514 - binary_accuracy: 1.0000
    Epoch 2419/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3513 - binary_accuracy: 1.0000
    Epoch 2420/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3512 - binary_accuracy: 1.0000
    Epoch 2421/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3511 - binary_accuracy: 1.0000
    Epoch 2422/7000
    1/1 [==============================] - 0s 18ms/step - loss: 0.3511 - binary_accuracy: 1.0000
    Epoch 2423/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3510 - binary_accuracy: 1.0000
    Epoch 2424/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3509 - binary_accuracy: 1.0000
    Epoch 2425/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3509 - binary_accuracy: 1.0000
    Epoch 2426/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3508 - binary_accuracy: 1.0000
    Epoch 2427/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3507 - binary_accuracy: 1.0000
    Epoch 2428/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3507 - binary_accuracy: 1.0000
    Epoch 2429/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3506 - binary_accuracy: 1.0000
    Epoch 2430/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3505 - binary_accuracy: 1.0000
    Epoch 2431/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3504 - binary_accuracy: 1.0000
    Epoch 2432/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.3504 - binary_accuracy: 1.0000
    Epoch 2433/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3503 - binary_accuracy: 1.0000
    Epoch 2434/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3502 - binary_accuracy: 1.0000
    Epoch 2435/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3502 - binary_accuracy: 1.0000
    Epoch 2436/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3501 - binary_accuracy: 1.0000
    Epoch 2437/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3500 - binary_accuracy: 1.0000
    Epoch 2438/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3500 - binary_accuracy: 1.0000
    Epoch 2439/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3499 - binary_accuracy: 1.0000
    Epoch 2440/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3498 - binary_accuracy: 1.0000
    Epoch 2441/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3497 - binary_accuracy: 1.0000
    Epoch 2442/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.3497 - binary_accuracy: 1.0000
    Epoch 2443/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3496 - binary_accuracy: 1.0000
    Epoch 2444/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3495 - binary_accuracy: 1.0000
    Epoch 2445/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3495 - binary_accuracy: 1.0000
    Epoch 2446/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3494 - binary_accuracy: 1.0000
    Epoch 2447/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3493 - binary_accuracy: 1.0000
    Epoch 2448/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3493 - binary_accuracy: 1.0000
    Epoch 2449/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3492 - binary_accuracy: 1.0000
    Epoch 2450/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3491 - binary_accuracy: 1.0000
    Epoch 2451/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3491 - binary_accuracy: 1.0000
    Epoch 2452/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3490 - binary_accuracy: 1.0000
    Epoch 2453/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3489 - binary_accuracy: 1.0000
    Epoch 2454/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3488 - binary_accuracy: 1.0000
    Epoch 2455/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3488 - binary_accuracy: 1.0000
    Epoch 2456/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3487 - binary_accuracy: 1.0000
    Epoch 2457/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3486 - binary_accuracy: 1.0000
    Epoch 2458/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3486 - binary_accuracy: 1.0000
    Epoch 2459/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3485 - binary_accuracy: 1.0000
    Epoch 2460/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3484 - binary_accuracy: 1.0000
    Epoch 2461/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3484 - binary_accuracy: 1.0000
    Epoch 2462/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3483 - binary_accuracy: 1.0000
    Epoch 2463/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3482 - binary_accuracy: 1.0000
    Epoch 2464/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3482 - binary_accuracy: 1.0000
    Epoch 2465/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3481 - binary_accuracy: 1.0000
    Epoch 2466/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3480 - binary_accuracy: 1.0000
    Epoch 2467/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3480 - binary_accuracy: 1.0000
    Epoch 2468/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3479 - binary_accuracy: 1.0000
    Epoch 2469/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3478 - binary_accuracy: 1.0000
    Epoch 2470/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3477 - binary_accuracy: 1.0000
    Epoch 2471/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3477 - binary_accuracy: 1.0000
    Epoch 2472/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.3476 - binary_accuracy: 1.0000
    Epoch 2473/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3475 - binary_accuracy: 1.0000
    Epoch 2474/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3475 - binary_accuracy: 1.0000
    Epoch 2475/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3474 - binary_accuracy: 1.0000
    Epoch 2476/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3473 - binary_accuracy: 1.0000
    Epoch 2477/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3473 - binary_accuracy: 1.0000
    Epoch 2478/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3472 - binary_accuracy: 1.0000
    Epoch 2479/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3471 - binary_accuracy: 1.0000
    Epoch 2480/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3471 - binary_accuracy: 1.0000
    Epoch 2481/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3470 - binary_accuracy: 1.0000
    Epoch 2482/7000
    1/1 [==============================] - 0s 27ms/step - loss: 0.3469 - binary_accuracy: 1.0000
    Epoch 2483/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3469 - binary_accuracy: 1.0000
    Epoch 2484/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3468 - binary_accuracy: 1.0000
    Epoch 2485/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3467 - binary_accuracy: 1.0000
    Epoch 2486/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3467 - binary_accuracy: 1.0000
    Epoch 2487/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3466 - binary_accuracy: 1.0000
    Epoch 2488/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3465 - binary_accuracy: 1.0000
    Epoch 2489/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3465 - binary_accuracy: 1.0000
    Epoch 2490/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3464 - binary_accuracy: 1.0000
    Epoch 2491/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3463 - binary_accuracy: 1.0000
    Epoch 2492/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3462 - binary_accuracy: 1.0000
    Epoch 2493/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3462 - binary_accuracy: 1.0000
    Epoch 2494/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3461 - binary_accuracy: 1.0000
    Epoch 2495/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3460 - binary_accuracy: 1.0000
    Epoch 2496/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3460 - binary_accuracy: 1.0000
    Epoch 2497/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3459 - binary_accuracy: 1.0000
    Epoch 2498/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3458 - binary_accuracy: 1.0000
    Epoch 2499/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3458 - binary_accuracy: 1.0000
    Epoch 2500/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3457 - binary_accuracy: 1.0000
    Epoch 2501/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3456 - binary_accuracy: 1.0000
    Epoch 2502/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3456 - binary_accuracy: 1.0000
    Epoch 2503/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3455 - binary_accuracy: 1.0000
    Epoch 2504/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3454 - binary_accuracy: 1.0000
    Epoch 2505/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3454 - binary_accuracy: 1.0000
    Epoch 2506/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3453 - binary_accuracy: 1.0000
    Epoch 2507/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3452 - binary_accuracy: 1.0000
    Epoch 2508/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3452 - binary_accuracy: 1.0000
    Epoch 2509/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3451 - binary_accuracy: 1.0000
    Epoch 2510/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3450 - binary_accuracy: 1.0000
    Epoch 2511/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3450 - binary_accuracy: 1.0000
    Epoch 2512/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3449 - binary_accuracy: 1.0000
    Epoch 2513/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3448 - binary_accuracy: 1.0000
    Epoch 2514/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3448 - binary_accuracy: 1.0000
    Epoch 2515/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3447 - binary_accuracy: 1.0000
    Epoch 2516/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3446 - binary_accuracy: 1.0000
    Epoch 2517/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3446 - binary_accuracy: 1.0000
    Epoch 2518/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3445 - binary_accuracy: 1.0000
    Epoch 2519/7000
    1/1 [==============================] - 0s 16ms/step - loss: 0.3444 - binary_accuracy: 1.0000
    Epoch 2520/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3444 - binary_accuracy: 1.0000
    Epoch 2521/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3443 - binary_accuracy: 1.0000
    Epoch 2522/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3442 - binary_accuracy: 1.0000
    Epoch 2523/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3442 - binary_accuracy: 1.0000
    Epoch 2524/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3441 - binary_accuracy: 1.0000
    Epoch 2525/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3440 - binary_accuracy: 1.0000
    Epoch 2526/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3440 - binary_accuracy: 1.0000
    Epoch 2527/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3439 - binary_accuracy: 1.0000
    Epoch 2528/7000
    1/1 [==============================] - 0s 57ms/step - loss: 0.3438 - binary_accuracy: 1.0000
    Epoch 2529/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3438 - binary_accuracy: 1.0000
    Epoch 2530/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3437 - binary_accuracy: 1.0000
    Epoch 2531/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3436 - binary_accuracy: 1.0000
    Epoch 2532/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3436 - binary_accuracy: 1.0000
    Epoch 2533/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3435 - binary_accuracy: 1.0000
    Epoch 2534/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3434 - binary_accuracy: 1.0000
    Epoch 2535/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3434 - binary_accuracy: 1.0000
    Epoch 2536/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3433 - binary_accuracy: 1.0000
    Epoch 2537/7000
    1/1 [==============================] - 0s 17ms/step - loss: 0.3432 - binary_accuracy: 1.0000
    Epoch 2538/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3432 - binary_accuracy: 1.0000
    Epoch 2539/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3431 - binary_accuracy: 1.0000
    Epoch 2540/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3430 - binary_accuracy: 1.0000
    Epoch 2541/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3430 - binary_accuracy: 1.0000
    Epoch 2542/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3429 - binary_accuracy: 1.0000
    Epoch 2543/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3428 - binary_accuracy: 1.0000
    Epoch 2544/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3428 - binary_accuracy: 1.0000
    Epoch 2545/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3427 - binary_accuracy: 1.0000
    Epoch 2546/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3426 - binary_accuracy: 1.0000
    Epoch 2547/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3426 - binary_accuracy: 1.0000
    Epoch 2548/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3425 - binary_accuracy: 1.0000
    Epoch 2549/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3424 - binary_accuracy: 1.0000
    Epoch 2550/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3424 - binary_accuracy: 1.0000
    Epoch 2551/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3423 - binary_accuracy: 1.0000
    Epoch 2552/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3422 - binary_accuracy: 1.0000
    Epoch 2553/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3422 - binary_accuracy: 1.0000
    Epoch 2554/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3421 - binary_accuracy: 1.0000
    Epoch 2555/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3420 - binary_accuracy: 1.0000
    Epoch 2556/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3420 - binary_accuracy: 1.0000
    Epoch 2557/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3419 - binary_accuracy: 1.0000
    Epoch 2558/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3418 - binary_accuracy: 1.0000
    Epoch 2559/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3418 - binary_accuracy: 1.0000
    Epoch 2560/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3417 - binary_accuracy: 1.0000
    Epoch 2561/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3416 - binary_accuracy: 1.0000
    Epoch 2562/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3416 - binary_accuracy: 1.0000
    Epoch 2563/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3415 - binary_accuracy: 1.0000
    Epoch 2564/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3414 - binary_accuracy: 1.0000
    Epoch 2565/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3414 - binary_accuracy: 1.0000
    Epoch 2566/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3413 - binary_accuracy: 1.0000
    Epoch 2567/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.3412 - binary_accuracy: 1.0000
    Epoch 2568/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3412 - binary_accuracy: 1.0000
    Epoch 2569/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3411 - binary_accuracy: 1.0000
    Epoch 2570/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3410 - binary_accuracy: 1.0000
    Epoch 2571/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3410 - binary_accuracy: 1.0000
    Epoch 2572/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3409 - binary_accuracy: 1.0000
    Epoch 2573/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3408 - binary_accuracy: 1.0000
    Epoch 2574/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3408 - binary_accuracy: 1.0000
    Epoch 2575/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3407 - binary_accuracy: 1.0000
    Epoch 2576/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.3406 - binary_accuracy: 1.0000
    Epoch 2577/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3406 - binary_accuracy: 1.0000
    Epoch 2578/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3405 - binary_accuracy: 1.0000
    Epoch 2579/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3405 - binary_accuracy: 1.0000
    Epoch 2580/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3404 - binary_accuracy: 1.0000
    Epoch 2581/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3403 - binary_accuracy: 1.0000
    Epoch 2582/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3403 - binary_accuracy: 1.0000
    Epoch 2583/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3402 - binary_accuracy: 1.0000
    Epoch 2584/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3401 - binary_accuracy: 1.0000
    Epoch 2585/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3401 - binary_accuracy: 1.0000
    Epoch 2586/7000
    1/1 [==============================] - 0s 18ms/step - loss: 0.3400 - binary_accuracy: 1.0000
    Epoch 2587/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3399 - binary_accuracy: 1.0000
    Epoch 2588/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3399 - binary_accuracy: 1.0000
    Epoch 2589/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3398 - binary_accuracy: 1.0000
    Epoch 2590/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3397 - binary_accuracy: 1.0000
    Epoch 2591/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3397 - binary_accuracy: 1.0000
    Epoch 2592/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3396 - binary_accuracy: 1.0000
    Epoch 2593/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3395 - binary_accuracy: 1.0000
    Epoch 2594/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3395 - binary_accuracy: 1.0000
    Epoch 2595/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3394 - binary_accuracy: 1.0000
    Epoch 2596/7000
    1/1 [==============================] - 0s 24ms/step - loss: 0.3393 - binary_accuracy: 1.0000
    Epoch 2597/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3393 - binary_accuracy: 1.0000
    Epoch 2598/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3392 - binary_accuracy: 1.0000
    Epoch 2599/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3392 - binary_accuracy: 1.0000
    Epoch 2600/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3391 - binary_accuracy: 1.0000
    Epoch 2601/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3390 - binary_accuracy: 1.0000
    Epoch 2602/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3390 - binary_accuracy: 1.0000
    Epoch 2603/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3389 - binary_accuracy: 1.0000
    Epoch 2604/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3388 - binary_accuracy: 1.0000
    Epoch 2605/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3388 - binary_accuracy: 1.0000
    Epoch 2606/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3387 - binary_accuracy: 1.0000
    Epoch 2607/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3386 - binary_accuracy: 1.0000
    Epoch 2608/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3386 - binary_accuracy: 1.0000
    Epoch 2609/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3385 - binary_accuracy: 1.0000
    Epoch 2610/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3384 - binary_accuracy: 1.0000
    Epoch 2611/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3384 - binary_accuracy: 1.0000
    Epoch 2612/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3383 - binary_accuracy: 1.0000
    Epoch 2613/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3382 - binary_accuracy: 1.0000
    Epoch 2614/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3382 - binary_accuracy: 1.0000
    Epoch 2615/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3381 - binary_accuracy: 1.0000
    Epoch 2616/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3381 - binary_accuracy: 1.0000
    Epoch 2617/7000
    1/1 [==============================] - 0s 31ms/step - loss: 0.3380 - binary_accuracy: 1.0000
    Epoch 2618/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3379 - binary_accuracy: 1.0000
    Epoch 2619/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3379 - binary_accuracy: 1.0000
    Epoch 2620/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3378 - binary_accuracy: 1.0000
    Epoch 2621/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3377 - binary_accuracy: 1.0000
    Epoch 2622/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3377 - binary_accuracy: 1.0000
    Epoch 2623/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3376 - binary_accuracy: 1.0000
    Epoch 2624/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3375 - binary_accuracy: 1.0000
    Epoch 2625/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3375 - binary_accuracy: 1.0000
    Epoch 2626/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3374 - binary_accuracy: 1.0000
    Epoch 2627/7000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3373 - binary_accuracy: 1.0000
    Epoch 2628/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3373 - binary_accuracy: 1.0000
    Epoch 2629/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3372 - binary_accuracy: 1.0000
    Epoch 2630/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3372 - binary_accuracy: 1.0000
    Epoch 2631/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3371 - binary_accuracy: 1.0000
    Epoch 2632/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3370 - binary_accuracy: 1.0000
    Epoch 2633/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3370 - binary_accuracy: 1.0000
    Epoch 2634/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3369 - binary_accuracy: 1.0000
    Epoch 2635/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3368 - binary_accuracy: 1.0000
    Epoch 2636/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3368 - binary_accuracy: 1.0000
    Epoch 2637/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.3367 - binary_accuracy: 1.0000
    Epoch 2638/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3366 - binary_accuracy: 1.0000
    Epoch 2639/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3366 - binary_accuracy: 1.0000
    Epoch 2640/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3365 - binary_accuracy: 1.0000
    Epoch 2641/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3365 - binary_accuracy: 1.0000
    Epoch 2642/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3364 - binary_accuracy: 1.0000
    Epoch 2643/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3363 - binary_accuracy: 1.0000
    Epoch 2644/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3363 - binary_accuracy: 1.0000
    Epoch 2645/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3362 - binary_accuracy: 1.0000
    Epoch 2646/7000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3361 - binary_accuracy: 1.0000
    Epoch 2647/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3361 - binary_accuracy: 1.0000
    Epoch 2648/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3360 - binary_accuracy: 1.0000
    Epoch 2649/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3359 - binary_accuracy: 1.0000
    Epoch 2650/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3359 - binary_accuracy: 1.0000
    Epoch 2651/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3358 - binary_accuracy: 1.0000
    Epoch 2652/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3358 - binary_accuracy: 1.0000
    Epoch 2653/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3357 - binary_accuracy: 1.0000
    Epoch 2654/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3356 - binary_accuracy: 1.0000
    Epoch 2655/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3356 - binary_accuracy: 1.0000
    Epoch 2656/7000
    1/1 [==============================] - 0s 46ms/step - loss: 0.3355 - binary_accuracy: 1.0000
    Epoch 2657/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3354 - binary_accuracy: 1.0000
    Epoch 2658/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3354 - binary_accuracy: 1.0000
    Epoch 2659/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3353 - binary_accuracy: 1.0000
    Epoch 2660/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3353 - binary_accuracy: 1.0000
    Epoch 2661/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3352 - binary_accuracy: 1.0000
    Epoch 2662/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3351 - binary_accuracy: 1.0000
    Epoch 2663/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3351 - binary_accuracy: 1.0000
    Epoch 2664/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3350 - binary_accuracy: 1.0000
    Epoch 2665/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3349 - binary_accuracy: 1.0000
    Epoch 2666/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3349 - binary_accuracy: 1.0000
    Epoch 2667/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3348 - binary_accuracy: 1.0000
    Epoch 2668/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3347 - binary_accuracy: 1.0000
    Epoch 2669/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3347 - binary_accuracy: 1.0000
    Epoch 2670/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3346 - binary_accuracy: 1.0000
    Epoch 2671/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3346 - binary_accuracy: 1.0000
    Epoch 2672/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3345 - binary_accuracy: 1.0000
    Epoch 2673/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3344 - binary_accuracy: 1.0000
    Epoch 2674/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3344 - binary_accuracy: 1.0000
    Epoch 2675/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3343 - binary_accuracy: 1.0000
    Epoch 2676/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3342 - binary_accuracy: 1.0000
    Epoch 2677/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3342 - binary_accuracy: 1.0000
    Epoch 2678/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3341 - binary_accuracy: 1.0000
    Epoch 2679/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3341 - binary_accuracy: 1.0000
    Epoch 2680/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3340 - binary_accuracy: 1.0000
    Epoch 2681/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3339 - binary_accuracy: 1.0000
    Epoch 2682/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3339 - binary_accuracy: 1.0000
    Epoch 2683/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3338 - binary_accuracy: 1.0000
    Epoch 2684/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3337 - binary_accuracy: 1.0000
    Epoch 2685/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.3337 - binary_accuracy: 1.0000
    Epoch 2686/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3336 - binary_accuracy: 1.0000
    Epoch 2687/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3336 - binary_accuracy: 1.0000
    Epoch 2688/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3335 - binary_accuracy: 1.0000
    Epoch 2689/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3334 - binary_accuracy: 1.0000
    Epoch 2690/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3334 - binary_accuracy: 1.0000
    Epoch 2691/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3333 - binary_accuracy: 1.0000
    Epoch 2692/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3332 - binary_accuracy: 1.0000
    Epoch 2693/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3332 - binary_accuracy: 1.0000
    Epoch 2694/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3331 - binary_accuracy: 1.0000
    Epoch 2695/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3331 - binary_accuracy: 1.0000
    Epoch 2696/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3330 - binary_accuracy: 1.0000
    Epoch 2697/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3329 - binary_accuracy: 1.0000
    Epoch 2698/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3329 - binary_accuracy: 1.0000
    Epoch 2699/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3328 - binary_accuracy: 1.0000
    Epoch 2700/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3327 - binary_accuracy: 1.0000
    Epoch 2701/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3327 - binary_accuracy: 1.0000
    Epoch 2702/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3326 - binary_accuracy: 1.0000
    Epoch 2703/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3326 - binary_accuracy: 1.0000
    Epoch 2704/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3325 - binary_accuracy: 1.0000
    Epoch 2705/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3324 - binary_accuracy: 1.0000
    Epoch 2706/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3324 - binary_accuracy: 1.0000
    Epoch 2707/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3323 - binary_accuracy: 1.0000
    Epoch 2708/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3323 - binary_accuracy: 1.0000
    Epoch 2709/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3322 - binary_accuracy: 1.0000
    Epoch 2710/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3321 - binary_accuracy: 1.0000
    Epoch 2711/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3321 - binary_accuracy: 1.0000
    Epoch 2712/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3320 - binary_accuracy: 1.0000
    Epoch 2713/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3319 - binary_accuracy: 1.0000
    Epoch 2714/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3319 - binary_accuracy: 1.0000
    Epoch 2715/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.3318 - binary_accuracy: 1.0000
    Epoch 2716/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3318 - binary_accuracy: 1.0000
    Epoch 2717/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3317 - binary_accuracy: 1.0000
    Epoch 2718/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3316 - binary_accuracy: 1.0000
    Epoch 2719/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3316 - binary_accuracy: 1.0000
    Epoch 2720/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3315 - binary_accuracy: 1.0000
    Epoch 2721/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3315 - binary_accuracy: 1.0000
    Epoch 2722/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3314 - binary_accuracy: 1.0000
    Epoch 2723/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3313 - binary_accuracy: 1.0000
    Epoch 2724/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3313 - binary_accuracy: 1.0000
    Epoch 2725/7000
    1/1 [==============================] - 0s 15ms/step - loss: 0.3312 - binary_accuracy: 1.0000
    Epoch 2726/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3311 - binary_accuracy: 1.0000
    Epoch 2727/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3311 - binary_accuracy: 1.0000
    Epoch 2728/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3310 - binary_accuracy: 1.0000
    Epoch 2729/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3310 - binary_accuracy: 1.0000
    Epoch 2730/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3309 - binary_accuracy: 1.0000
    Epoch 2731/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3308 - binary_accuracy: 1.0000
    Epoch 2732/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3308 - binary_accuracy: 1.0000
    Epoch 2733/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3307 - binary_accuracy: 1.0000
    Epoch 2734/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3307 - binary_accuracy: 1.0000
    Epoch 2735/7000
    1/1 [==============================] - 0s 15ms/step - loss: 0.3306 - binary_accuracy: 1.0000
    Epoch 2736/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3305 - binary_accuracy: 1.0000
    Epoch 2737/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3305 - binary_accuracy: 1.0000
    Epoch 2738/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3304 - binary_accuracy: 1.0000
    Epoch 2739/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3303 - binary_accuracy: 1.0000
    Epoch 2740/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3303 - binary_accuracy: 1.0000
    Epoch 2741/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3302 - binary_accuracy: 1.0000
    Epoch 2742/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3302 - binary_accuracy: 1.0000
    Epoch 2743/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3301 - binary_accuracy: 1.0000
    Epoch 2744/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3300 - binary_accuracy: 1.0000
    Epoch 2745/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3300 - binary_accuracy: 1.0000
    Epoch 2746/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3299 - binary_accuracy: 1.0000
    Epoch 2747/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3299 - binary_accuracy: 1.0000
    Epoch 2748/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3298 - binary_accuracy: 1.0000
    Epoch 2749/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3297 - binary_accuracy: 1.0000
    Epoch 2750/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3297 - binary_accuracy: 1.0000
    Epoch 2751/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3296 - binary_accuracy: 1.0000
    Epoch 2752/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3296 - binary_accuracy: 1.0000
    Epoch 2753/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3295 - binary_accuracy: 1.0000
    Epoch 2754/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3294 - binary_accuracy: 1.0000
    Epoch 2755/7000
    1/1 [==============================] - 0s 22ms/step - loss: 0.3294 - binary_accuracy: 1.0000
    Epoch 2756/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3293 - binary_accuracy: 1.0000
    Epoch 2757/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3293 - binary_accuracy: 1.0000
    Epoch 2758/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3292 - binary_accuracy: 1.0000
    Epoch 2759/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3291 - binary_accuracy: 1.0000
    Epoch 2760/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3291 - binary_accuracy: 1.0000
    Epoch 2761/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3290 - binary_accuracy: 1.0000
    Epoch 2762/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3290 - binary_accuracy: 1.0000
    Epoch 2763/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3289 - binary_accuracy: 1.0000
    Epoch 2764/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3288 - binary_accuracy: 1.0000
    Epoch 2765/7000
    1/1 [==============================] - 0s 22ms/step - loss: 0.3288 - binary_accuracy: 1.0000
    Epoch 2766/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3287 - binary_accuracy: 1.0000
    Epoch 2767/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3286 - binary_accuracy: 1.0000
    Epoch 2768/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3286 - binary_accuracy: 1.0000
    Epoch 2769/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3285 - binary_accuracy: 1.0000
    Epoch 2770/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3285 - binary_accuracy: 1.0000
    Epoch 2771/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3284 - binary_accuracy: 1.0000
    Epoch 2772/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3283 - binary_accuracy: 1.0000
    Epoch 2773/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3283 - binary_accuracy: 1.0000
    Epoch 2774/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3282 - binary_accuracy: 1.0000
    Epoch 2775/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3282 - binary_accuracy: 1.0000
    Epoch 2776/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3281 - binary_accuracy: 1.0000
    Epoch 2777/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3280 - binary_accuracy: 1.0000
    Epoch 2778/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3280 - binary_accuracy: 1.0000
    Epoch 2779/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3279 - binary_accuracy: 1.0000
    Epoch 2780/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3279 - binary_accuracy: 1.0000
    Epoch 2781/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3278 - binary_accuracy: 1.0000
    Epoch 2782/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3277 - binary_accuracy: 1.0000
    Epoch 2783/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3277 - binary_accuracy: 1.0000
    Epoch 2784/7000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3276 - binary_accuracy: 1.0000
    Epoch 2785/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3276 - binary_accuracy: 1.0000
    Epoch 2786/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3275 - binary_accuracy: 1.0000
    Epoch 2787/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3274 - binary_accuracy: 1.0000
    Epoch 2788/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3274 - binary_accuracy: 1.0000
    Epoch 2789/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3273 - binary_accuracy: 1.0000
    Epoch 2790/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3273 - binary_accuracy: 1.0000
    Epoch 2791/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3272 - binary_accuracy: 1.0000
    Epoch 2792/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3271 - binary_accuracy: 1.0000
    Epoch 2793/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3271 - binary_accuracy: 1.0000
    Epoch 2794/7000
    1/1 [==============================] - 0s 16ms/step - loss: 0.3270 - binary_accuracy: 1.0000
    Epoch 2795/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3270 - binary_accuracy: 1.0000
    Epoch 2796/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3269 - binary_accuracy: 1.0000
    Epoch 2797/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3268 - binary_accuracy: 1.0000
    Epoch 2798/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3268 - binary_accuracy: 1.0000
    Epoch 2799/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3267 - binary_accuracy: 1.0000
    Epoch 2800/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3267 - binary_accuracy: 1.0000
    Epoch 2801/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3266 - binary_accuracy: 1.0000
    Epoch 2802/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3266 - binary_accuracy: 1.0000
    Epoch 2803/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3265 - binary_accuracy: 1.0000
    Epoch 2804/7000
    1/1 [==============================] - 0s 20ms/step - loss: 0.3264 - binary_accuracy: 1.0000
    Epoch 2805/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3264 - binary_accuracy: 1.0000
    Epoch 2806/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3263 - binary_accuracy: 1.0000
    Epoch 2807/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3263 - binary_accuracy: 1.0000
    Epoch 2808/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3262 - binary_accuracy: 1.0000
    Epoch 2809/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3261 - binary_accuracy: 1.0000
    Epoch 2810/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3261 - binary_accuracy: 1.0000
    Epoch 2811/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3260 - binary_accuracy: 1.0000
    Epoch 2812/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3260 - binary_accuracy: 1.0000
    Epoch 2813/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3259 - binary_accuracy: 1.0000
    Epoch 2814/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3258 - binary_accuracy: 1.0000
    Epoch 2815/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3258 - binary_accuracy: 1.0000
    Epoch 2816/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3257 - binary_accuracy: 1.0000
    Epoch 2817/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3257 - binary_accuracy: 1.0000
    Epoch 2818/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3256 - binary_accuracy: 1.0000
    Epoch 2819/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3255 - binary_accuracy: 1.0000
    Epoch 2820/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3255 - binary_accuracy: 1.0000
    Epoch 2821/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3254 - binary_accuracy: 1.0000
    Epoch 2822/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3254 - binary_accuracy: 1.0000
    Epoch 2823/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3253 - binary_accuracy: 1.0000
    Epoch 2824/7000
    1/1 [==============================] - 0s 24ms/step - loss: 0.3252 - binary_accuracy: 1.0000
    Epoch 2825/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3252 - binary_accuracy: 1.0000
    Epoch 2826/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3251 - binary_accuracy: 1.0000
    Epoch 2827/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3251 - binary_accuracy: 1.0000
    Epoch 2828/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3250 - binary_accuracy: 1.0000
    Epoch 2829/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3250 - binary_accuracy: 1.0000
    Epoch 2830/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3249 - binary_accuracy: 1.0000
    Epoch 2831/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3248 - binary_accuracy: 1.0000
    Epoch 2832/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3248 - binary_accuracy: 1.0000
    Epoch 2833/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3247 - binary_accuracy: 1.0000
    Epoch 2834/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3247 - binary_accuracy: 1.0000
    Epoch 2835/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3246 - binary_accuracy: 1.0000
    Epoch 2836/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3245 - binary_accuracy: 1.0000
    Epoch 2837/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3245 - binary_accuracy: 1.0000
    Epoch 2838/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3244 - binary_accuracy: 1.0000
    Epoch 2839/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3244 - binary_accuracy: 1.0000
    Epoch 2840/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3243 - binary_accuracy: 1.0000
    Epoch 2841/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3243 - binary_accuracy: 1.0000
    Epoch 2842/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3242 - binary_accuracy: 1.0000
    Epoch 2843/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3241 - binary_accuracy: 1.0000
    Epoch 2844/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.3241 - binary_accuracy: 1.0000
    Epoch 2845/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3240 - binary_accuracy: 1.0000
    Epoch 2846/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3240 - binary_accuracy: 1.0000
    Epoch 2847/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3239 - binary_accuracy: 1.0000
    Epoch 2848/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3238 - binary_accuracy: 1.0000
    Epoch 2849/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3238 - binary_accuracy: 1.0000
    Epoch 2850/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3237 - binary_accuracy: 1.0000
    Epoch 2851/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3237 - binary_accuracy: 1.0000
    Epoch 2852/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3236 - binary_accuracy: 1.0000
    Epoch 2853/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3235 - binary_accuracy: 1.0000
    Epoch 2854/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.3235 - binary_accuracy: 1.0000
    Epoch 2855/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3234 - binary_accuracy: 1.0000
    Epoch 2856/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3234 - binary_accuracy: 1.0000
    Epoch 2857/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3233 - binary_accuracy: 1.0000
    Epoch 2858/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3233 - binary_accuracy: 1.0000
    Epoch 2859/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3232 - binary_accuracy: 1.0000
    Epoch 2860/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3231 - binary_accuracy: 1.0000
    Epoch 2861/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3231 - binary_accuracy: 1.0000
    Epoch 2862/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3230 - binary_accuracy: 1.0000
    Epoch 2863/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3230 - binary_accuracy: 1.0000
    Epoch 2864/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3229 - binary_accuracy: 1.0000
    Epoch 2865/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3229 - binary_accuracy: 1.0000
    Epoch 2866/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3228 - binary_accuracy: 1.0000
    Epoch 2867/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3227 - binary_accuracy: 1.0000
    Epoch 2868/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3227 - binary_accuracy: 1.0000
    Epoch 2869/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3226 - binary_accuracy: 1.0000
    Epoch 2870/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3226 - binary_accuracy: 1.0000
    Epoch 2871/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3225 - binary_accuracy: 1.0000
    Epoch 2872/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3224 - binary_accuracy: 1.0000
    Epoch 2873/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3224 - binary_accuracy: 1.0000
    Epoch 2874/7000
    1/1 [==============================] - 0s 28ms/step - loss: 0.3223 - binary_accuracy: 1.0000
    Epoch 2875/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3223 - binary_accuracy: 1.0000
    Epoch 2876/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3222 - binary_accuracy: 1.0000
    Epoch 2877/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3222 - binary_accuracy: 1.0000
    Epoch 2878/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3221 - binary_accuracy: 1.0000
    Epoch 2879/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3220 - binary_accuracy: 1.0000
    Epoch 2880/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3220 - binary_accuracy: 1.0000
    Epoch 2881/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3219 - binary_accuracy: 1.0000
    Epoch 2882/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3219 - binary_accuracy: 1.0000
    Epoch 2883/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3218 - binary_accuracy: 1.0000
    Epoch 2884/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.3218 - binary_accuracy: 1.0000
    Epoch 2885/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3217 - binary_accuracy: 1.0000
    Epoch 2886/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3216 - binary_accuracy: 1.0000
    Epoch 2887/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3216 - binary_accuracy: 1.0000
    Epoch 2888/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3215 - binary_accuracy: 1.0000
    Epoch 2889/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3215 - binary_accuracy: 1.0000
    Epoch 2890/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3214 - binary_accuracy: 1.0000
    Epoch 2891/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3214 - binary_accuracy: 1.0000
    Epoch 2892/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3213 - binary_accuracy: 1.0000
    Epoch 2893/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3212 - binary_accuracy: 1.0000
    Epoch 2894/7000
    1/1 [==============================] - 0s 20ms/step - loss: 0.3212 - binary_accuracy: 1.0000
    Epoch 2895/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3211 - binary_accuracy: 1.0000
    Epoch 2896/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3211 - binary_accuracy: 1.0000
    Epoch 2897/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3210 - binary_accuracy: 1.0000
    Epoch 2898/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3210 - binary_accuracy: 1.0000
    Epoch 2899/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3209 - binary_accuracy: 1.0000
    Epoch 2900/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3208 - binary_accuracy: 1.0000
    Epoch 2901/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3208 - binary_accuracy: 1.0000
    Epoch 2902/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3207 - binary_accuracy: 1.0000
    Epoch 2903/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3207 - binary_accuracy: 1.0000
    Epoch 2904/7000
    1/1 [==============================] - 0s 26ms/step - loss: 0.3206 - binary_accuracy: 1.0000
    Epoch 2905/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3206 - binary_accuracy: 1.0000
    Epoch 2906/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3205 - binary_accuracy: 1.0000
    Epoch 2907/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3204 - binary_accuracy: 1.0000
    Epoch 2908/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3204 - binary_accuracy: 1.0000
    Epoch 2909/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3203 - binary_accuracy: 1.0000
    Epoch 2910/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3203 - binary_accuracy: 1.0000
    Epoch 2911/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3202 - binary_accuracy: 1.0000
    Epoch 2912/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3202 - binary_accuracy: 1.0000
    Epoch 2913/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3201 - binary_accuracy: 1.0000
    Epoch 2914/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3200 - binary_accuracy: 1.0000
    Epoch 2915/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3200 - binary_accuracy: 1.0000
    Epoch 2916/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3199 - binary_accuracy: 1.0000
    Epoch 2917/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3199 - binary_accuracy: 1.0000
    Epoch 2918/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3198 - binary_accuracy: 1.0000
    Epoch 2919/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3198 - binary_accuracy: 1.0000
    Epoch 2920/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3197 - binary_accuracy: 1.0000
    Epoch 2921/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3196 - binary_accuracy: 1.0000
    Epoch 2922/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3196 - binary_accuracy: 1.0000
    Epoch 2923/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3195 - binary_accuracy: 1.0000
    Epoch 2924/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3195 - binary_accuracy: 1.0000
    Epoch 2925/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3194 - binary_accuracy: 1.0000
    Epoch 2926/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3194 - binary_accuracy: 1.0000
    Epoch 2927/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3193 - binary_accuracy: 1.0000
    Epoch 2928/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3192 - binary_accuracy: 1.0000
    Epoch 2929/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3192 - binary_accuracy: 1.0000
    Epoch 2930/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3191 - binary_accuracy: 1.0000
    Epoch 2931/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3191 - binary_accuracy: 1.0000
    Epoch 2932/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3190 - binary_accuracy: 1.0000
    Epoch 2933/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3190 - binary_accuracy: 1.0000
    Epoch 2934/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3189 - binary_accuracy: 1.0000
    Epoch 2935/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3188 - binary_accuracy: 1.0000
    Epoch 2936/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3188 - binary_accuracy: 1.0000
    Epoch 2937/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3187 - binary_accuracy: 1.0000
    Epoch 2938/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3187 - binary_accuracy: 1.0000
    Epoch 2939/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3186 - binary_accuracy: 1.0000
    Epoch 2940/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3186 - binary_accuracy: 1.0000
    Epoch 2941/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3185 - binary_accuracy: 1.0000
    Epoch 2942/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3185 - binary_accuracy: 1.0000
    Epoch 2943/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3184 - binary_accuracy: 1.0000
    Epoch 2944/7000
    1/1 [==============================] - 0s 44ms/step - loss: 0.3183 - binary_accuracy: 1.0000
    Epoch 2945/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3183 - binary_accuracy: 1.0000
    Epoch 2946/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3182 - binary_accuracy: 1.0000
    Epoch 2947/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3182 - binary_accuracy: 1.0000
    Epoch 2948/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3181 - binary_accuracy: 1.0000
    Epoch 2949/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3181 - binary_accuracy: 1.0000
    Epoch 2950/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3180 - binary_accuracy: 1.0000
    Epoch 2951/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3179 - binary_accuracy: 1.0000
    Epoch 2952/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3179 - binary_accuracy: 1.0000
    Epoch 2953/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3178 - binary_accuracy: 1.0000
    Epoch 2954/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3178 - binary_accuracy: 1.0000
    Epoch 2955/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3177 - binary_accuracy: 1.0000
    Epoch 2956/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3177 - binary_accuracy: 1.0000
    Epoch 2957/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3176 - binary_accuracy: 1.0000
    Epoch 2958/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3176 - binary_accuracy: 1.0000
    Epoch 2959/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3175 - binary_accuracy: 1.0000
    Epoch 2960/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3174 - binary_accuracy: 1.0000
    Epoch 2961/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3174 - binary_accuracy: 1.0000
    Epoch 2962/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3173 - binary_accuracy: 1.0000
    Epoch 2963/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3173 - binary_accuracy: 1.0000
    Epoch 2964/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.3172 - binary_accuracy: 1.0000
    Epoch 2965/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3172 - binary_accuracy: 1.0000
    Epoch 2966/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3171 - binary_accuracy: 1.0000
    Epoch 2967/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3171 - binary_accuracy: 1.0000
    Epoch 2968/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3170 - binary_accuracy: 1.0000
    Epoch 2969/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3169 - binary_accuracy: 1.0000
    Epoch 2970/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3169 - binary_accuracy: 1.0000
    Epoch 2971/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3168 - binary_accuracy: 1.0000
    Epoch 2972/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3168 - binary_accuracy: 1.0000
    Epoch 2973/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3167 - binary_accuracy: 1.0000
    Epoch 2974/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3167 - binary_accuracy: 1.0000
    Epoch 2975/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3166 - binary_accuracy: 1.0000
    Epoch 2976/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3166 - binary_accuracy: 1.0000
    Epoch 2977/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3165 - binary_accuracy: 1.0000
    Epoch 2978/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3164 - binary_accuracy: 1.0000
    Epoch 2979/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3164 - binary_accuracy: 1.0000
    Epoch 2980/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3163 - binary_accuracy: 1.0000
    Epoch 2981/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3163 - binary_accuracy: 1.0000
    Epoch 2982/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3162 - binary_accuracy: 1.0000
    Epoch 2983/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3162 - binary_accuracy: 1.0000
    Epoch 2984/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3161 - binary_accuracy: 1.0000
    Epoch 2985/7000
    1/1 [==============================] - 0s 26ms/step - loss: 0.3161 - binary_accuracy: 1.0000
    Epoch 2986/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3160 - binary_accuracy: 1.0000
    Epoch 2987/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3159 - binary_accuracy: 1.0000
    Epoch 2988/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3159 - binary_accuracy: 1.0000
    Epoch 2989/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3158 - binary_accuracy: 1.0000
    Epoch 2990/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3158 - binary_accuracy: 1.0000
    Epoch 2991/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3157 - binary_accuracy: 1.0000
    Epoch 2992/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3157 - binary_accuracy: 1.0000
    Epoch 2993/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3156 - binary_accuracy: 1.0000
    Epoch 2994/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3156 - binary_accuracy: 1.0000
    Epoch 2995/7000
    1/1 [==============================] - 0s 27ms/step - loss: 0.3155 - binary_accuracy: 1.0000
    Epoch 2996/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3154 - binary_accuracy: 1.0000
    Epoch 2997/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3154 - binary_accuracy: 1.0000
    Epoch 2998/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3153 - binary_accuracy: 1.0000
    Epoch 2999/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3153 - binary_accuracy: 1.0000
    Epoch 3000/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3152 - binary_accuracy: 1.0000
    Epoch 3001/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3152 - binary_accuracy: 1.0000
    Epoch 3002/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.3151 - binary_accuracy: 1.0000
    Epoch 3003/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3151 - binary_accuracy: 1.0000
    Epoch 3004/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3150 - binary_accuracy: 1.0000
    Epoch 3005/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3150 - binary_accuracy: 1.0000
    Epoch 3006/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3149 - binary_accuracy: 1.0000
    Epoch 3007/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3148 - binary_accuracy: 1.0000
    Epoch 3008/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3148 - binary_accuracy: 1.0000
    Epoch 3009/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3147 - binary_accuracy: 1.0000
    Epoch 3010/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3147 - binary_accuracy: 1.0000
    Epoch 3011/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3146 - binary_accuracy: 1.0000
    Epoch 3012/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3146 - binary_accuracy: 1.0000
    Epoch 3013/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3145 - binary_accuracy: 1.0000
    Epoch 3014/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3145 - binary_accuracy: 1.0000
    Epoch 3015/7000
    1/1 [==============================] - 0s 29ms/step - loss: 0.3144 - binary_accuracy: 1.0000
    Epoch 3016/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3144 - binary_accuracy: 1.0000
    Epoch 3017/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3143 - binary_accuracy: 1.0000
    Epoch 3018/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3142 - binary_accuracy: 1.0000
    Epoch 3019/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3142 - binary_accuracy: 1.0000
    Epoch 3020/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3141 - binary_accuracy: 1.0000
    Epoch 3021/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3141 - binary_accuracy: 1.0000
    Epoch 3022/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3140 - binary_accuracy: 1.0000
    Epoch 3023/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3140 - binary_accuracy: 1.0000
    Epoch 3024/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3139 - binary_accuracy: 1.0000
    Epoch 3025/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.3139 - binary_accuracy: 1.0000
    Epoch 3026/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3138 - binary_accuracy: 1.0000
    Epoch 3027/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3138 - binary_accuracy: 1.0000
    Epoch 3028/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3137 - binary_accuracy: 1.0000
    Epoch 3029/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3136 - binary_accuracy: 1.0000
    Epoch 3030/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3136 - binary_accuracy: 1.0000
    Epoch 3031/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3135 - binary_accuracy: 1.0000
    Epoch 3032/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3135 - binary_accuracy: 1.0000
    Epoch 3033/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3134 - binary_accuracy: 1.0000
    Epoch 3034/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3134 - binary_accuracy: 1.0000
    Epoch 3035/7000
    1/1 [==============================] - 0s 40ms/step - loss: 0.3133 - binary_accuracy: 1.0000
    Epoch 3036/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3133 - binary_accuracy: 1.0000
    Epoch 3037/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3132 - binary_accuracy: 1.0000
    Epoch 3038/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3132 - binary_accuracy: 1.0000
    Epoch 3039/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3131 - binary_accuracy: 1.0000
    Epoch 3040/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3130 - binary_accuracy: 1.0000
    Epoch 3041/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3130 - binary_accuracy: 1.0000
    Epoch 3042/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3129 - binary_accuracy: 1.0000
    Epoch 3043/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3129 - binary_accuracy: 1.0000
    Epoch 3044/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3128 - binary_accuracy: 1.0000
    Epoch 3045/7000
    1/1 [==============================] - 0s 30ms/step - loss: 0.3128 - binary_accuracy: 1.0000
    Epoch 3046/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3127 - binary_accuracy: 1.0000
    Epoch 3047/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3127 - binary_accuracy: 1.0000
    Epoch 3048/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3126 - binary_accuracy: 1.0000
    Epoch 3049/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3126 - binary_accuracy: 1.0000
    Epoch 3050/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3125 - binary_accuracy: 1.0000
    Epoch 3051/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3125 - binary_accuracy: 1.0000
    Epoch 3052/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3124 - binary_accuracy: 1.0000
    Epoch 3053/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3123 - binary_accuracy: 1.0000
    Epoch 3054/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3123 - binary_accuracy: 1.0000
    Epoch 3055/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3122 - binary_accuracy: 1.0000
    Epoch 3056/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3122 - binary_accuracy: 1.0000
    Epoch 3057/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3121 - binary_accuracy: 1.0000
    Epoch 3058/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3121 - binary_accuracy: 1.0000
    Epoch 3059/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3120 - binary_accuracy: 1.0000
    Epoch 3060/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3120 - binary_accuracy: 1.0000
    Epoch 3061/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3119 - binary_accuracy: 1.0000
    Epoch 3062/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3119 - binary_accuracy: 1.0000
    Epoch 3063/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3118 - binary_accuracy: 1.0000
    Epoch 3064/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3118 - binary_accuracy: 1.0000
    Epoch 3065/7000
    1/1 [==============================] - 0s 47ms/step - loss: 0.3117 - binary_accuracy: 1.0000
    Epoch 3066/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3116 - binary_accuracy: 1.0000
    Epoch 3067/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3116 - binary_accuracy: 1.0000
    Epoch 3068/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3115 - binary_accuracy: 1.0000
    Epoch 3069/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3115 - binary_accuracy: 1.0000
    Epoch 3070/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3114 - binary_accuracy: 1.0000
    Epoch 3071/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3114 - binary_accuracy: 1.0000
    Epoch 3072/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3113 - binary_accuracy: 1.0000
    Epoch 3073/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3113 - binary_accuracy: 1.0000
    Epoch 3074/7000
    1/1 [==============================] - 0s 74ms/step - loss: 0.3112 - binary_accuracy: 1.0000
    Epoch 3075/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3112 - binary_accuracy: 1.0000
    Epoch 3076/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3111 - binary_accuracy: 1.0000
    Epoch 3077/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3111 - binary_accuracy: 1.0000
    Epoch 3078/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3110 - binary_accuracy: 1.0000
    Epoch 3079/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3110 - binary_accuracy: 1.0000
    Epoch 3080/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3109 - binary_accuracy: 1.0000
    Epoch 3081/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3108 - binary_accuracy: 1.0000
    Epoch 3082/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3108 - binary_accuracy: 1.0000
    Epoch 3083/7000
    1/1 [==============================] - 0s 64ms/step - loss: 0.3107 - binary_accuracy: 1.0000
    Epoch 3084/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3107 - binary_accuracy: 1.0000
    Epoch 3085/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3106 - binary_accuracy: 1.0000
    Epoch 3086/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3106 - binary_accuracy: 1.0000
    Epoch 3087/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3105 - binary_accuracy: 1.0000
    Epoch 3088/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3105 - binary_accuracy: 1.0000
    Epoch 3089/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3104 - binary_accuracy: 1.0000
    Epoch 3090/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3104 - binary_accuracy: 1.0000
    Epoch 3091/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3103 - binary_accuracy: 1.0000
    Epoch 3092/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.3103 - binary_accuracy: 1.0000
    Epoch 3093/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3102 - binary_accuracy: 1.0000
    Epoch 3094/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3102 - binary_accuracy: 1.0000
    Epoch 3095/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3101 - binary_accuracy: 1.0000
    Epoch 3096/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3100 - binary_accuracy: 1.0000
    Epoch 3097/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3100 - binary_accuracy: 1.0000
    Epoch 3098/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3099 - binary_accuracy: 1.0000
    Epoch 3099/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3099 - binary_accuracy: 1.0000
    Epoch 3100/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3098 - binary_accuracy: 1.0000
    Epoch 3101/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3098 - binary_accuracy: 1.0000
    Epoch 3102/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3097 - binary_accuracy: 1.0000
    Epoch 3103/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3097 - binary_accuracy: 1.0000
    Epoch 3104/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3096 - binary_accuracy: 1.0000
    Epoch 3105/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3096 - binary_accuracy: 1.0000
    Epoch 3106/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3095 - binary_accuracy: 1.0000
    Epoch 3107/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3095 - binary_accuracy: 1.0000
    Epoch 3108/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3094 - binary_accuracy: 1.0000
    Epoch 3109/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3094 - binary_accuracy: 1.0000
    Epoch 3110/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3093 - binary_accuracy: 1.0000
    Epoch 3111/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3093 - binary_accuracy: 1.0000
    Epoch 3112/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3092 - binary_accuracy: 1.0000
    Epoch 3113/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3092 - binary_accuracy: 1.0000
    Epoch 3114/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3091 - binary_accuracy: 1.0000
    Epoch 3115/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3090 - binary_accuracy: 1.0000
    Epoch 3116/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3090 - binary_accuracy: 1.0000
    Epoch 3117/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3089 - binary_accuracy: 1.0000
    Epoch 3118/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3089 - binary_accuracy: 1.0000
    Epoch 3119/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3088 - binary_accuracy: 1.0000
    Epoch 3120/7000
    1/1 [==============================] - 0s 18ms/step - loss: 0.3088 - binary_accuracy: 1.0000
    Epoch 3121/7000
    1/1 [==============================] - 0s 39ms/step - loss: 0.3087 - binary_accuracy: 1.0000
    Epoch 3122/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3087 - binary_accuracy: 1.0000
    Epoch 3123/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3086 - binary_accuracy: 1.0000
    Epoch 3124/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3086 - binary_accuracy: 1.0000
    Epoch 3125/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3085 - binary_accuracy: 1.0000
    Epoch 3126/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3085 - binary_accuracy: 1.0000
    Epoch 3127/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3084 - binary_accuracy: 1.0000
    Epoch 3128/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3084 - binary_accuracy: 1.0000
    Epoch 3129/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3083 - binary_accuracy: 1.0000
    Epoch 3130/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3083 - binary_accuracy: 1.0000
    Epoch 3131/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.3082 - binary_accuracy: 1.0000
    Epoch 3132/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3082 - binary_accuracy: 1.0000
    Epoch 3133/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3081 - binary_accuracy: 1.0000
    Epoch 3134/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3080 - binary_accuracy: 1.0000
    Epoch 3135/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3080 - binary_accuracy: 1.0000
    Epoch 3136/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3079 - binary_accuracy: 1.0000
    Epoch 3137/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3079 - binary_accuracy: 1.0000
    Epoch 3138/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3078 - binary_accuracy: 1.0000
    Epoch 3139/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3078 - binary_accuracy: 1.0000
    Epoch 3140/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3077 - binary_accuracy: 1.0000
    Epoch 3141/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3077 - binary_accuracy: 1.0000
    Epoch 3142/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3076 - binary_accuracy: 1.0000
    Epoch 3143/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3076 - binary_accuracy: 1.0000
    Epoch 3144/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3075 - binary_accuracy: 1.0000
    Epoch 3145/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3075 - binary_accuracy: 1.0000
    Epoch 3146/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3074 - binary_accuracy: 1.0000
    Epoch 3147/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3074 - binary_accuracy: 1.0000
    Epoch 3148/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3073 - binary_accuracy: 1.0000
    Epoch 3149/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3073 - binary_accuracy: 1.0000
    Epoch 3150/7000
    1/1 [==============================] - 0s 29ms/step - loss: 0.3072 - binary_accuracy: 1.0000
    Epoch 3151/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3072 - binary_accuracy: 1.0000
    Epoch 3152/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3071 - binary_accuracy: 1.0000
    Epoch 3153/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3071 - binary_accuracy: 1.0000
    Epoch 3154/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3070 - binary_accuracy: 1.0000
    Epoch 3155/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3070 - binary_accuracy: 1.0000
    Epoch 3156/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3069 - binary_accuracy: 1.0000
    Epoch 3157/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3069 - binary_accuracy: 1.0000
    Epoch 3158/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3068 - binary_accuracy: 1.0000
    Epoch 3159/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3068 - binary_accuracy: 1.0000
    Epoch 3160/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.3067 - binary_accuracy: 1.0000
    Epoch 3161/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3066 - binary_accuracy: 1.0000
    Epoch 3162/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3066 - binary_accuracy: 1.0000
    Epoch 3163/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3065 - binary_accuracy: 1.0000
    Epoch 3164/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3065 - binary_accuracy: 1.0000
    Epoch 3165/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3064 - binary_accuracy: 1.0000
    Epoch 3166/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3064 - binary_accuracy: 1.0000
    Epoch 3167/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3063 - binary_accuracy: 1.0000
    Epoch 3168/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3063 - binary_accuracy: 1.0000
    Epoch 3169/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3062 - binary_accuracy: 1.0000
    Epoch 3170/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3062 - binary_accuracy: 1.0000
    Epoch 3171/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3061 - binary_accuracy: 1.0000
    Epoch 3172/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3061 - binary_accuracy: 1.0000
    Epoch 3173/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3060 - binary_accuracy: 1.0000
    Epoch 3174/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3060 - binary_accuracy: 1.0000
    Epoch 3175/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3059 - binary_accuracy: 1.0000
    Epoch 3176/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3059 - binary_accuracy: 1.0000
    Epoch 3177/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3058 - binary_accuracy: 1.0000
    Epoch 3178/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3058 - binary_accuracy: 1.0000
    Epoch 3179/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3057 - binary_accuracy: 1.0000
    Epoch 3180/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3057 - binary_accuracy: 1.0000
    Epoch 3181/7000
    1/1 [==============================] - 0s 42ms/step - loss: 0.3056 - binary_accuracy: 1.0000
    Epoch 3182/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3056 - binary_accuracy: 1.0000
    Epoch 3183/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3055 - binary_accuracy: 1.0000
    Epoch 3184/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3055 - binary_accuracy: 1.0000
    Epoch 3185/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3054 - binary_accuracy: 1.0000
    Epoch 3186/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3054 - binary_accuracy: 1.0000
    Epoch 3187/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3053 - binary_accuracy: 1.0000
    Epoch 3188/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3053 - binary_accuracy: 1.0000
    Epoch 3189/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3052 - binary_accuracy: 1.0000
    Epoch 3190/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3052 - binary_accuracy: 1.0000
    Epoch 3191/7000
    1/1 [==============================] - 0s 22ms/step - loss: 0.3051 - binary_accuracy: 1.0000
    Epoch 3192/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3051 - binary_accuracy: 1.0000
    Epoch 3193/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3050 - binary_accuracy: 1.0000
    Epoch 3194/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3050 - binary_accuracy: 1.0000
    Epoch 3195/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3049 - binary_accuracy: 1.0000
    Epoch 3196/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3048 - binary_accuracy: 1.0000
    Epoch 3197/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3048 - binary_accuracy: 1.0000
    Epoch 3198/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3047 - binary_accuracy: 1.0000
    Epoch 3199/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3047 - binary_accuracy: 1.0000
    Epoch 3200/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3046 - binary_accuracy: 1.0000
    Epoch 3201/7000
    1/1 [==============================] - 0s 34ms/step - loss: 0.3046 - binary_accuracy: 1.0000
    Epoch 3202/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3045 - binary_accuracy: 1.0000
    Epoch 3203/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3045 - binary_accuracy: 1.0000
    Epoch 3204/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3044 - binary_accuracy: 1.0000
    Epoch 3205/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3044 - binary_accuracy: 1.0000
    Epoch 3206/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3043 - binary_accuracy: 1.0000
    Epoch 3207/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3043 - binary_accuracy: 1.0000
    Epoch 3208/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3042 - binary_accuracy: 1.0000
    Epoch 3209/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3042 - binary_accuracy: 1.0000
    Epoch 3210/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3041 - binary_accuracy: 1.0000
    Epoch 3211/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3041 - binary_accuracy: 1.0000
    Epoch 3212/7000
    1/1 [==============================] - 0s 35ms/step - loss: 0.3040 - binary_accuracy: 1.0000
    Epoch 3213/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3040 - binary_accuracy: 1.0000
    Epoch 3214/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3039 - binary_accuracy: 1.0000
    Epoch 3215/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3039 - binary_accuracy: 1.0000
    Epoch 3216/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3038 - binary_accuracy: 1.0000
    Epoch 3217/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3038 - binary_accuracy: 1.0000
    Epoch 3218/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3037 - binary_accuracy: 1.0000
    Epoch 3219/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3037 - binary_accuracy: 1.0000
    Epoch 3220/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3036 - binary_accuracy: 1.0000
    Epoch 3221/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3036 - binary_accuracy: 1.0000
    Epoch 3222/7000
    1/1 [==============================] - 0s 30ms/step - loss: 0.3035 - binary_accuracy: 1.0000
    Epoch 3223/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3035 - binary_accuracy: 1.0000
    Epoch 3224/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3034 - binary_accuracy: 1.0000
    Epoch 3225/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3034 - binary_accuracy: 1.0000
    Epoch 3226/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3033 - binary_accuracy: 1.0000
    Epoch 3227/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3033 - binary_accuracy: 1.0000
    Epoch 3228/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3032 - binary_accuracy: 1.0000
    Epoch 3229/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3032 - binary_accuracy: 1.0000
    Epoch 3230/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3031 - binary_accuracy: 1.0000
    Epoch 3231/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3031 - binary_accuracy: 1.0000
    Epoch 3232/7000
    1/1 [==============================] - 0s 41ms/step - loss: 0.3030 - binary_accuracy: 1.0000
    Epoch 3233/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3030 - binary_accuracy: 1.0000
    Epoch 3234/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3029 - binary_accuracy: 1.0000
    Epoch 3235/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3029 - binary_accuracy: 1.0000
    Epoch 3236/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.3028 - binary_accuracy: 1.0000
    Epoch 3237/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3028 - binary_accuracy: 1.0000
    Epoch 3238/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3027 - binary_accuracy: 1.0000
    Epoch 3239/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3027 - binary_accuracy: 1.0000
    Epoch 3240/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3026 - binary_accuracy: 1.0000
    Epoch 3241/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3026 - binary_accuracy: 1.0000
    Epoch 3242/7000
    1/1 [==============================] - 0s 20ms/step - loss: 0.3025 - binary_accuracy: 1.0000
    Epoch 3243/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3025 - binary_accuracy: 1.0000
    Epoch 3244/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3024 - binary_accuracy: 1.0000
    Epoch 3245/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3024 - binary_accuracy: 1.0000
    Epoch 3246/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3023 - binary_accuracy: 1.0000
    Epoch 3247/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3023 - binary_accuracy: 1.0000
    Epoch 3248/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3022 - binary_accuracy: 1.0000
    Epoch 3249/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3022 - binary_accuracy: 1.0000
    Epoch 3250/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3021 - binary_accuracy: 1.0000
    Epoch 3251/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3021 - binary_accuracy: 1.0000
    Epoch 3252/7000
    1/1 [==============================] - 0s 45ms/step - loss: 0.3020 - binary_accuracy: 1.0000
    Epoch 3253/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3020 - binary_accuracy: 1.0000
    Epoch 3254/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3019 - binary_accuracy: 1.0000
    Epoch 3255/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3019 - binary_accuracy: 1.0000
    Epoch 3256/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3018 - binary_accuracy: 1.0000
    Epoch 3257/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3018 - binary_accuracy: 1.0000
    Epoch 3258/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3017 - binary_accuracy: 1.0000
    Epoch 3259/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3017 - binary_accuracy: 1.0000
    Epoch 3260/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3016 - binary_accuracy: 1.0000
    Epoch 3261/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3016 - binary_accuracy: 1.0000
    Epoch 3262/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3015 - binary_accuracy: 1.0000
    Epoch 3263/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3015 - binary_accuracy: 1.0000
    Epoch 3264/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3014 - binary_accuracy: 1.0000
    Epoch 3265/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3014 - binary_accuracy: 1.0000
    Epoch 3266/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3013 - binary_accuracy: 1.0000
    Epoch 3267/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3013 - binary_accuracy: 1.0000
    Epoch 3268/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3012 - binary_accuracy: 1.0000
    Epoch 3269/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3012 - binary_accuracy: 1.0000
    Epoch 3270/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.3011 - binary_accuracy: 1.0000
    Epoch 3271/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.3011 - binary_accuracy: 1.0000
    Epoch 3272/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3010 - binary_accuracy: 1.0000
    Epoch 3273/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3010 - binary_accuracy: 1.0000
    Epoch 3274/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3009 - binary_accuracy: 1.0000
    Epoch 3275/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3009 - binary_accuracy: 1.0000
    Epoch 3276/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3008 - binary_accuracy: 1.0000
    Epoch 3277/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3008 - binary_accuracy: 1.0000
    Epoch 3278/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3007 - binary_accuracy: 1.0000
    Epoch 3279/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3007 - binary_accuracy: 1.0000
    Epoch 3280/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3006 - binary_accuracy: 1.0000
    Epoch 3281/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.3006 - binary_accuracy: 1.0000
    Epoch 3282/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3005 - binary_accuracy: 1.0000
    Epoch 3283/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3005 - binary_accuracy: 1.0000
    Epoch 3284/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3004 - binary_accuracy: 1.0000
    Epoch 3285/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3004 - binary_accuracy: 1.0000
    Epoch 3286/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3003 - binary_accuracy: 1.0000
    Epoch 3287/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.3003 - binary_accuracy: 1.0000
    Epoch 3288/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3002 - binary_accuracy: 1.0000
    Epoch 3289/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3002 - binary_accuracy: 1.0000
    Epoch 3290/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.3001 - binary_accuracy: 1.0000
    Epoch 3291/7000
    1/1 [==============================] - 0s 50ms/step - loss: 0.3001 - binary_accuracy: 1.0000
    Epoch 3292/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.3000 - binary_accuracy: 1.0000
    Epoch 3293/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.3000 - binary_accuracy: 1.0000
    Epoch 3294/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2999 - binary_accuracy: 1.0000
    Epoch 3295/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2999 - binary_accuracy: 1.0000
    Epoch 3296/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2998 - binary_accuracy: 1.0000
    Epoch 3297/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2998 - binary_accuracy: 1.0000
    Epoch 3298/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2997 - binary_accuracy: 1.0000
    Epoch 3299/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2997 - binary_accuracy: 1.0000
    Epoch 3300/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2996 - binary_accuracy: 1.0000
    Epoch 3301/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2996 - binary_accuracy: 1.0000
    Epoch 3302/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2995 - binary_accuracy: 1.0000
    Epoch 3303/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2995 - binary_accuracy: 1.0000
    Epoch 3304/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2994 - binary_accuracy: 1.0000
    Epoch 3305/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2994 - binary_accuracy: 1.0000
    Epoch 3306/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2993 - binary_accuracy: 1.0000
    Epoch 3307/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2993 - binary_accuracy: 1.0000
    Epoch 3308/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2993 - binary_accuracy: 1.0000
    Epoch 3309/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2992 - binary_accuracy: 1.0000
    Epoch 3310/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2992 - binary_accuracy: 1.0000
    Epoch 3311/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2991 - binary_accuracy: 1.0000
    Epoch 3312/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2991 - binary_accuracy: 1.0000
    Epoch 3313/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2990 - binary_accuracy: 1.0000
    Epoch 3314/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2990 - binary_accuracy: 1.0000
    Epoch 3315/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2989 - binary_accuracy: 1.0000
    Epoch 3316/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2989 - binary_accuracy: 1.0000
    Epoch 3317/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2988 - binary_accuracy: 1.0000
    Epoch 3318/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2988 - binary_accuracy: 1.0000
    Epoch 3319/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2987 - binary_accuracy: 1.0000
    Epoch 3320/7000
    1/1 [==============================] - 0s 20ms/step - loss: 0.2987 - binary_accuracy: 1.0000
    Epoch 3321/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2986 - binary_accuracy: 1.0000
    Epoch 3322/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2986 - binary_accuracy: 1.0000
    Epoch 3323/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2985 - binary_accuracy: 1.0000
    Epoch 3324/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2985 - binary_accuracy: 1.0000
    Epoch 3325/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2984 - binary_accuracy: 1.0000
    Epoch 3326/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2984 - binary_accuracy: 1.0000
    Epoch 3327/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2983 - binary_accuracy: 1.0000
    Epoch 3328/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2983 - binary_accuracy: 1.0000
    Epoch 3329/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2982 - binary_accuracy: 1.0000
    Epoch 3330/7000
    1/1 [==============================] - 0s 30ms/step - loss: 0.2982 - binary_accuracy: 1.0000
    Epoch 3331/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2981 - binary_accuracy: 1.0000
    Epoch 3332/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2981 - binary_accuracy: 1.0000
    Epoch 3333/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2980 - binary_accuracy: 1.0000
    Epoch 3334/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2980 - binary_accuracy: 1.0000
    Epoch 3335/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2979 - binary_accuracy: 1.0000
    Epoch 3336/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2979 - binary_accuracy: 1.0000
    Epoch 3337/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2978 - binary_accuracy: 1.0000
    Epoch 3338/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2978 - binary_accuracy: 1.0000
    Epoch 3339/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2977 - binary_accuracy: 1.0000
    Epoch 3340/7000
    1/1 [==============================] - 0s 34ms/step - loss: 0.2977 - binary_accuracy: 1.0000
    Epoch 3341/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2976 - binary_accuracy: 1.0000
    Epoch 3342/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2976 - binary_accuracy: 1.0000
    Epoch 3343/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2975 - binary_accuracy: 1.0000
    Epoch 3344/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2975 - binary_accuracy: 1.0000
    Epoch 3345/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2975 - binary_accuracy: 1.0000
    Epoch 3346/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2974 - binary_accuracy: 1.0000
    Epoch 3347/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2974 - binary_accuracy: 1.0000
    Epoch 3348/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2973 - binary_accuracy: 1.0000
    Epoch 3349/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2973 - binary_accuracy: 1.0000
    Epoch 3350/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2972 - binary_accuracy: 1.0000
    Epoch 3351/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2972 - binary_accuracy: 1.0000
    Epoch 3352/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2971 - binary_accuracy: 1.0000
    Epoch 3353/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2971 - binary_accuracy: 1.0000
    Epoch 3354/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2970 - binary_accuracy: 1.0000
    Epoch 3355/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2970 - binary_accuracy: 1.0000
    Epoch 3356/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2969 - binary_accuracy: 1.0000
    Epoch 3357/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2969 - binary_accuracy: 1.0000
    Epoch 3358/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2968 - binary_accuracy: 1.0000
    Epoch 3359/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2968 - binary_accuracy: 1.0000
    Epoch 3360/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.2967 - binary_accuracy: 1.0000
    Epoch 3361/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2967 - binary_accuracy: 1.0000
    Epoch 3362/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2966 - binary_accuracy: 1.0000
    Epoch 3363/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2966 - binary_accuracy: 1.0000
    Epoch 3364/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2965 - binary_accuracy: 1.0000
    Epoch 3365/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2965 - binary_accuracy: 1.0000
    Epoch 3366/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2964 - binary_accuracy: 1.0000
    Epoch 3367/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2964 - binary_accuracy: 1.0000
    Epoch 3368/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2963 - binary_accuracy: 1.0000
    Epoch 3369/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2963 - binary_accuracy: 1.0000
    Epoch 3370/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2962 - binary_accuracy: 1.0000
    Epoch 3371/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2962 - binary_accuracy: 1.0000
    Epoch 3372/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2962 - binary_accuracy: 1.0000
    Epoch 3373/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2961 - binary_accuracy: 1.0000
    Epoch 3374/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2961 - binary_accuracy: 1.0000
    Epoch 3375/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2960 - binary_accuracy: 1.0000
    Epoch 3376/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2960 - binary_accuracy: 1.0000
    Epoch 3377/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2959 - binary_accuracy: 1.0000
    Epoch 3378/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2959 - binary_accuracy: 1.0000
    Epoch 3379/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2958 - binary_accuracy: 1.0000
    Epoch 3380/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2958 - binary_accuracy: 1.0000
    Epoch 3381/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2957 - binary_accuracy: 1.0000
    Epoch 3382/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2957 - binary_accuracy: 1.0000
    Epoch 3383/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2956 - binary_accuracy: 1.0000
    Epoch 3384/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2956 - binary_accuracy: 1.0000
    Epoch 3385/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2955 - binary_accuracy: 1.0000
    Epoch 3386/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2955 - binary_accuracy: 1.0000
    Epoch 3387/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2954 - binary_accuracy: 1.0000
    Epoch 3388/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2954 - binary_accuracy: 1.0000
    Epoch 3389/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2953 - binary_accuracy: 1.0000
    Epoch 3390/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.2953 - binary_accuracy: 1.0000
    Epoch 3391/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2952 - binary_accuracy: 1.0000
    Epoch 3392/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2952 - binary_accuracy: 1.0000
    Epoch 3393/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2952 - binary_accuracy: 1.0000
    Epoch 3394/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2951 - binary_accuracy: 1.0000
    Epoch 3395/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2951 - binary_accuracy: 1.0000
    Epoch 3396/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2950 - binary_accuracy: 1.0000
    Epoch 3397/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2950 - binary_accuracy: 1.0000
    Epoch 3398/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2949 - binary_accuracy: 1.0000
    Epoch 3399/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2949 - binary_accuracy: 1.0000
    Epoch 3400/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.2948 - binary_accuracy: 1.0000
    Epoch 3401/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2948 - binary_accuracy: 1.0000
    Epoch 3402/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2947 - binary_accuracy: 1.0000
    Epoch 3403/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2947 - binary_accuracy: 1.0000
    Epoch 3404/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2946 - binary_accuracy: 1.0000
    Epoch 3405/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2946 - binary_accuracy: 1.0000
    Epoch 3406/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2945 - binary_accuracy: 1.0000
    Epoch 3407/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2945 - binary_accuracy: 1.0000
    Epoch 3408/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2944 - binary_accuracy: 1.0000
    Epoch 3409/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2944 - binary_accuracy: 1.0000
    Epoch 3410/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2943 - binary_accuracy: 1.0000
    Epoch 3411/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2943 - binary_accuracy: 1.0000
    Epoch 3412/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2943 - binary_accuracy: 1.0000
    Epoch 3413/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2942 - binary_accuracy: 1.0000
    Epoch 3414/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2942 - binary_accuracy: 1.0000
    Epoch 3415/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2941 - binary_accuracy: 1.0000
    Epoch 3416/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2941 - binary_accuracy: 1.0000
    Epoch 3417/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2940 - binary_accuracy: 1.0000
    Epoch 3418/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2940 - binary_accuracy: 1.0000
    Epoch 3419/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2939 - binary_accuracy: 1.0000
    Epoch 3420/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2939 - binary_accuracy: 1.0000
    Epoch 3421/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2938 - binary_accuracy: 1.0000
    Epoch 3422/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2938 - binary_accuracy: 1.0000
    Epoch 3423/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2937 - binary_accuracy: 1.0000
    Epoch 3424/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2937 - binary_accuracy: 1.0000
    Epoch 3425/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2936 - binary_accuracy: 1.0000
    Epoch 3426/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2936 - binary_accuracy: 1.0000
    Epoch 3427/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2935 - binary_accuracy: 1.0000
    Epoch 3428/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2935 - binary_accuracy: 1.0000
    Epoch 3429/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2935 - binary_accuracy: 1.0000
    Epoch 3430/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2934 - binary_accuracy: 1.0000
    Epoch 3431/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2934 - binary_accuracy: 1.0000
    Epoch 3432/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2933 - binary_accuracy: 1.0000
    Epoch 3433/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2933 - binary_accuracy: 1.0000
    Epoch 3434/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2932 - binary_accuracy: 1.0000
    Epoch 3435/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2932 - binary_accuracy: 1.0000
    Epoch 3436/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2931 - binary_accuracy: 1.0000
    Epoch 3437/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2931 - binary_accuracy: 1.0000
    Epoch 3438/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2930 - binary_accuracy: 1.0000
    Epoch 3439/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2930 - binary_accuracy: 1.0000
    Epoch 3440/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2929 - binary_accuracy: 1.0000
    Epoch 3441/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2929 - binary_accuracy: 1.0000
    Epoch 3442/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2928 - binary_accuracy: 1.0000
    Epoch 3443/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2928 - binary_accuracy: 1.0000
    Epoch 3444/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2928 - binary_accuracy: 1.0000
    Epoch 3445/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2927 - binary_accuracy: 1.0000
    Epoch 3446/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2927 - binary_accuracy: 1.0000
    Epoch 3447/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2926 - binary_accuracy: 1.0000
    Epoch 3448/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2926 - binary_accuracy: 1.0000
    Epoch 3449/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2925 - binary_accuracy: 1.0000
    Epoch 3450/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2925 - binary_accuracy: 1.0000
    Epoch 3451/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2924 - binary_accuracy: 1.0000
    Epoch 3452/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2924 - binary_accuracy: 1.0000
    Epoch 3453/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2923 - binary_accuracy: 1.0000
    Epoch 3454/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2923 - binary_accuracy: 1.0000
    Epoch 3455/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2922 - binary_accuracy: 1.0000
    Epoch 3456/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2922 - binary_accuracy: 1.0000
    Epoch 3457/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2921 - binary_accuracy: 1.0000
    Epoch 3458/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2921 - binary_accuracy: 1.0000
    Epoch 3459/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2921 - binary_accuracy: 1.0000
    Epoch 3460/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2920 - binary_accuracy: 1.0000
    Epoch 3461/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2920 - binary_accuracy: 1.0000
    Epoch 3462/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2919 - binary_accuracy: 1.0000
    Epoch 3463/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2919 - binary_accuracy: 1.0000
    Epoch 3464/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2918 - binary_accuracy: 1.0000
    Epoch 3465/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2918 - binary_accuracy: 1.0000
    Epoch 3466/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2917 - binary_accuracy: 1.0000
    Epoch 3467/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2917 - binary_accuracy: 1.0000
    Epoch 3468/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2916 - binary_accuracy: 1.0000
    Epoch 3469/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2916 - binary_accuracy: 1.0000
    Epoch 3470/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2915 - binary_accuracy: 1.0000
    Epoch 3471/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2915 - binary_accuracy: 1.0000
    Epoch 3472/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2915 - binary_accuracy: 1.0000
    Epoch 3473/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2914 - binary_accuracy: 1.0000
    Epoch 3474/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2914 - binary_accuracy: 1.0000
    Epoch 3475/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2913 - binary_accuracy: 1.0000
    Epoch 3476/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2913 - binary_accuracy: 1.0000
    Epoch 3477/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2912 - binary_accuracy: 1.0000
    Epoch 3478/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2912 - binary_accuracy: 1.0000
    Epoch 3479/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2911 - binary_accuracy: 1.0000
    Epoch 3480/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2911 - binary_accuracy: 1.0000
    Epoch 3481/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2910 - binary_accuracy: 1.0000
    Epoch 3482/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2910 - binary_accuracy: 1.0000
    Epoch 3483/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2909 - binary_accuracy: 1.0000
    Epoch 3484/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2909 - binary_accuracy: 1.0000
    Epoch 3485/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2909 - binary_accuracy: 1.0000
    Epoch 3486/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2908 - binary_accuracy: 1.0000
    Epoch 3487/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2908 - binary_accuracy: 1.0000
    Epoch 3488/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2907 - binary_accuracy: 1.0000
    Epoch 3489/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2907 - binary_accuracy: 1.0000
    Epoch 3490/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2906 - binary_accuracy: 1.0000
    Epoch 3491/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2906 - binary_accuracy: 1.0000
    Epoch 3492/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2905 - binary_accuracy: 1.0000
    Epoch 3493/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2905 - binary_accuracy: 1.0000
    Epoch 3494/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2904 - binary_accuracy: 1.0000
    Epoch 3495/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2904 - binary_accuracy: 1.0000
    Epoch 3496/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2903 - binary_accuracy: 1.0000
    Epoch 3497/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2903 - binary_accuracy: 1.0000
    Epoch 3498/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2903 - binary_accuracy: 1.0000
    Epoch 3499/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2902 - binary_accuracy: 1.0000
    Epoch 3500/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2902 - binary_accuracy: 1.0000
    Epoch 3501/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2901 - binary_accuracy: 1.0000
    Epoch 3502/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2901 - binary_accuracy: 1.0000
    Epoch 3503/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2900 - binary_accuracy: 1.0000
    Epoch 3504/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2900 - binary_accuracy: 1.0000
    Epoch 3505/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2899 - binary_accuracy: 1.0000
    Epoch 3506/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2899 - binary_accuracy: 1.0000
    Epoch 3507/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2898 - binary_accuracy: 1.0000
    Epoch 3508/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2898 - binary_accuracy: 1.0000
    Epoch 3509/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2898 - binary_accuracy: 1.0000
    Epoch 3510/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2897 - binary_accuracy: 1.0000
    Epoch 3511/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2897 - binary_accuracy: 1.0000
    Epoch 3512/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2896 - binary_accuracy: 1.0000
    Epoch 3513/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2896 - binary_accuracy: 1.0000
    Epoch 3514/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2895 - binary_accuracy: 1.0000
    Epoch 3515/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2895 - binary_accuracy: 1.0000
    Epoch 3516/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2894 - binary_accuracy: 1.0000
    Epoch 3517/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2894 - binary_accuracy: 1.0000
    Epoch 3518/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2893 - binary_accuracy: 1.0000
    Epoch 3519/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2893 - binary_accuracy: 1.0000
    Epoch 3520/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2893 - binary_accuracy: 1.0000
    Epoch 3521/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2892 - binary_accuracy: 1.0000
    Epoch 3522/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2892 - binary_accuracy: 1.0000
    Epoch 3523/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2891 - binary_accuracy: 1.0000
    Epoch 3524/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2891 - binary_accuracy: 1.0000
    Epoch 3525/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2890 - binary_accuracy: 1.0000
    Epoch 3526/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2890 - binary_accuracy: 1.0000
    Epoch 3527/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2889 - binary_accuracy: 1.0000
    Epoch 3528/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2889 - binary_accuracy: 1.0000
    Epoch 3529/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2888 - binary_accuracy: 1.0000
    Epoch 3530/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2888 - binary_accuracy: 1.0000
    Epoch 3531/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2888 - binary_accuracy: 1.0000
    Epoch 3532/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2887 - binary_accuracy: 1.0000
    Epoch 3533/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2887 - binary_accuracy: 1.0000
    Epoch 3534/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2886 - binary_accuracy: 1.0000
    Epoch 3535/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2886 - binary_accuracy: 1.0000
    Epoch 3536/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2885 - binary_accuracy: 1.0000
    Epoch 3537/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2885 - binary_accuracy: 1.0000
    Epoch 3538/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2884 - binary_accuracy: 1.0000
    Epoch 3539/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2884 - binary_accuracy: 1.0000
    Epoch 3540/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2883 - binary_accuracy: 1.0000
    Epoch 3541/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2883 - binary_accuracy: 1.0000
    Epoch 3542/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2883 - binary_accuracy: 1.0000
    Epoch 3543/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2882 - binary_accuracy: 1.0000
    Epoch 3544/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2882 - binary_accuracy: 1.0000
    Epoch 3545/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2881 - binary_accuracy: 1.0000
    Epoch 3546/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2881 - binary_accuracy: 1.0000
    Epoch 3547/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2880 - binary_accuracy: 1.0000
    Epoch 3548/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2880 - binary_accuracy: 1.0000
    Epoch 3549/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2879 - binary_accuracy: 1.0000
    Epoch 3550/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2879 - binary_accuracy: 1.0000
    Epoch 3551/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2879 - binary_accuracy: 1.0000
    Epoch 3552/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2878 - binary_accuracy: 1.0000
    Epoch 3553/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2878 - binary_accuracy: 1.0000
    Epoch 3554/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2877 - binary_accuracy: 1.0000
    Epoch 3555/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2877 - binary_accuracy: 1.0000
    Epoch 3556/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2876 - binary_accuracy: 1.0000
    Epoch 3557/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2876 - binary_accuracy: 1.0000
    Epoch 3558/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2875 - binary_accuracy: 1.0000
    Epoch 3559/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2875 - binary_accuracy: 1.0000
    Epoch 3560/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2874 - binary_accuracy: 1.0000
    Epoch 3561/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2874 - binary_accuracy: 1.0000
    Epoch 3562/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2874 - binary_accuracy: 1.0000
    Epoch 3563/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2873 - binary_accuracy: 1.0000
    Epoch 3564/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2873 - binary_accuracy: 1.0000
    Epoch 3565/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2872 - binary_accuracy: 1.0000
    Epoch 3566/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2872 - binary_accuracy: 1.0000
    Epoch 3567/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2871 - binary_accuracy: 1.0000
    Epoch 3568/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2871 - binary_accuracy: 1.0000
    Epoch 3569/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2870 - binary_accuracy: 1.0000
    Epoch 3570/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2870 - binary_accuracy: 1.0000
    Epoch 3571/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2870 - binary_accuracy: 1.0000
    Epoch 3572/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2869 - binary_accuracy: 1.0000
    Epoch 3573/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2869 - binary_accuracy: 1.0000
    Epoch 3574/7000
    1/1 [==============================] - 0s 17ms/step - loss: 0.2868 - binary_accuracy: 1.0000
    Epoch 3575/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2868 - binary_accuracy: 1.0000
    Epoch 3576/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2867 - binary_accuracy: 1.0000
    Epoch 3577/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2867 - binary_accuracy: 1.0000
    Epoch 3578/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2866 - binary_accuracy: 1.0000
    Epoch 3579/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2866 - binary_accuracy: 1.0000
    Epoch 3580/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2866 - binary_accuracy: 1.0000
    Epoch 3581/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2865 - binary_accuracy: 1.0000
    Epoch 3582/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2865 - binary_accuracy: 1.0000
    Epoch 3583/7000
    1/1 [==============================] - 0s 20ms/step - loss: 0.2864 - binary_accuracy: 1.0000
    Epoch 3584/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2864 - binary_accuracy: 1.0000
    Epoch 3585/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2863 - binary_accuracy: 1.0000
    Epoch 3586/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2863 - binary_accuracy: 1.0000
    Epoch 3587/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2862 - binary_accuracy: 1.0000
    Epoch 3588/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2862 - binary_accuracy: 1.0000
    Epoch 3589/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2862 - binary_accuracy: 1.0000
    Epoch 3590/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2861 - binary_accuracy: 1.0000
    Epoch 3591/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2861 - binary_accuracy: 1.0000
    Epoch 3592/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2860 - binary_accuracy: 1.0000
    Epoch 3593/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.2860 - binary_accuracy: 1.0000
    Epoch 3594/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2859 - binary_accuracy: 1.0000
    Epoch 3595/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2859 - binary_accuracy: 1.0000
    Epoch 3596/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2858 - binary_accuracy: 1.0000
    Epoch 3597/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2858 - binary_accuracy: 1.0000
    Epoch 3598/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2858 - binary_accuracy: 1.0000
    Epoch 3599/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2857 - binary_accuracy: 1.0000
    Epoch 3600/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2857 - binary_accuracy: 1.0000
    Epoch 3601/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2856 - binary_accuracy: 1.0000
    Epoch 3602/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2856 - binary_accuracy: 1.0000
    Epoch 3603/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.2855 - binary_accuracy: 1.0000
    Epoch 3604/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2855 - binary_accuracy: 1.0000
    Epoch 3605/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2854 - binary_accuracy: 1.0000
    Epoch 3606/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2854 - binary_accuracy: 1.0000
    Epoch 3607/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2854 - binary_accuracy: 1.0000
    Epoch 3608/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2853 - binary_accuracy: 1.0000
    Epoch 3609/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2853 - binary_accuracy: 1.0000
    Epoch 3610/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2852 - binary_accuracy: 1.0000
    Epoch 3611/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2852 - binary_accuracy: 1.0000
    Epoch 3612/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2851 - binary_accuracy: 1.0000
    Epoch 3613/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2851 - binary_accuracy: 1.0000
    Epoch 3614/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2851 - binary_accuracy: 1.0000
    Epoch 3615/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2850 - binary_accuracy: 1.0000
    Epoch 3616/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2850 - binary_accuracy: 1.0000
    Epoch 3617/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2849 - binary_accuracy: 1.0000
    Epoch 3618/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2849 - binary_accuracy: 1.0000
    Epoch 3619/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2848 - binary_accuracy: 1.0000
    Epoch 3620/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2848 - binary_accuracy: 1.0000
    Epoch 3621/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2847 - binary_accuracy: 1.0000
    Epoch 3622/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.2847 - binary_accuracy: 1.0000
    Epoch 3623/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2847 - binary_accuracy: 1.0000
    Epoch 3624/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2846 - binary_accuracy: 1.0000
    Epoch 3625/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2846 - binary_accuracy: 1.0000
    Epoch 3626/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2845 - binary_accuracy: 1.0000
    Epoch 3627/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2845 - binary_accuracy: 1.0000
    Epoch 3628/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2844 - binary_accuracy: 1.0000
    Epoch 3629/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2844 - binary_accuracy: 1.0000
    Epoch 3630/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2843 - binary_accuracy: 1.0000
    Epoch 3631/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2843 - binary_accuracy: 1.0000
    Epoch 3632/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2843 - binary_accuracy: 1.0000
    Epoch 3633/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2842 - binary_accuracy: 1.0000
    Epoch 3634/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2842 - binary_accuracy: 1.0000
    Epoch 3635/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2841 - binary_accuracy: 1.0000
    Epoch 3636/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2841 - binary_accuracy: 1.0000
    Epoch 3637/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2840 - binary_accuracy: 1.0000
    Epoch 3638/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2840 - binary_accuracy: 1.0000
    Epoch 3639/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2840 - binary_accuracy: 1.0000
    Epoch 3640/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2839 - binary_accuracy: 1.0000
    Epoch 3641/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2839 - binary_accuracy: 1.0000
    Epoch 3642/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2838 - binary_accuracy: 1.0000
    Epoch 3643/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2838 - binary_accuracy: 1.0000
    Epoch 3644/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2837 - binary_accuracy: 1.0000
    Epoch 3645/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2837 - binary_accuracy: 1.0000
    Epoch 3646/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2837 - binary_accuracy: 1.0000
    Epoch 3647/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2836 - binary_accuracy: 1.0000
    Epoch 3648/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2836 - binary_accuracy: 1.0000
    Epoch 3649/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2835 - binary_accuracy: 1.0000
    Epoch 3650/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.2835 - binary_accuracy: 1.0000
    Epoch 3651/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2834 - binary_accuracy: 1.0000
    Epoch 3652/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2834 - binary_accuracy: 1.0000
    Epoch 3653/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2833 - binary_accuracy: 1.0000
    Epoch 3654/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2833 - binary_accuracy: 1.0000
    Epoch 3655/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2833 - binary_accuracy: 1.0000
    Epoch 3656/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2832 - binary_accuracy: 1.0000
    Epoch 3657/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2832 - binary_accuracy: 1.0000
    Epoch 3658/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2831 - binary_accuracy: 1.0000
    Epoch 3659/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2831 - binary_accuracy: 1.0000
    Epoch 3660/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2830 - binary_accuracy: 1.0000
    Epoch 3661/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2830 - binary_accuracy: 1.0000
    Epoch 3662/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2830 - binary_accuracy: 1.0000
    Epoch 3663/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2829 - binary_accuracy: 1.0000
    Epoch 3664/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2829 - binary_accuracy: 1.0000
    Epoch 3665/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2828 - binary_accuracy: 1.0000
    Epoch 3666/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2828 - binary_accuracy: 1.0000
    Epoch 3667/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2827 - binary_accuracy: 1.0000
    Epoch 3668/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2827 - binary_accuracy: 1.0000
    Epoch 3669/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2827 - binary_accuracy: 1.0000
    Epoch 3670/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2826 - binary_accuracy: 1.0000
    Epoch 3671/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2826 - binary_accuracy: 1.0000
    Epoch 3672/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2825 - binary_accuracy: 1.0000
    Epoch 3673/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2825 - binary_accuracy: 1.0000
    Epoch 3674/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2824 - binary_accuracy: 1.0000
    Epoch 3675/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2824 - binary_accuracy: 1.0000
    Epoch 3676/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2824 - binary_accuracy: 1.0000
    Epoch 3677/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2823 - binary_accuracy: 1.0000
    Epoch 3678/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2823 - binary_accuracy: 1.0000
    Epoch 3679/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2822 - binary_accuracy: 1.0000
    Epoch 3680/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2822 - binary_accuracy: 1.0000
    Epoch 3681/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2821 - binary_accuracy: 1.0000
    Epoch 3682/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2821 - binary_accuracy: 1.0000
    Epoch 3683/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2821 - binary_accuracy: 1.0000
    Epoch 3684/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2820 - binary_accuracy: 1.0000
    Epoch 3685/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2820 - binary_accuracy: 1.0000
    Epoch 3686/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2819 - binary_accuracy: 1.0000
    Epoch 3687/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2819 - binary_accuracy: 1.0000
    Epoch 3688/7000
    1/1 [==============================] - 0s 21ms/step - loss: 0.2818 - binary_accuracy: 1.0000
    Epoch 3689/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2818 - binary_accuracy: 1.0000
    Epoch 3690/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2818 - binary_accuracy: 1.0000
    Epoch 3691/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2817 - binary_accuracy: 1.0000
    Epoch 3692/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2817 - binary_accuracy: 1.0000
    Epoch 3693/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2816 - binary_accuracy: 1.0000
    Epoch 3694/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2816 - binary_accuracy: 1.0000
    Epoch 3695/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2815 - binary_accuracy: 1.0000
    Epoch 3696/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2815 - binary_accuracy: 1.0000
    Epoch 3697/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2815 - binary_accuracy: 1.0000
    Epoch 3698/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2814 - binary_accuracy: 1.0000
    Epoch 3699/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2814 - binary_accuracy: 1.0000
    Epoch 3700/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2813 - binary_accuracy: 1.0000
    Epoch 3701/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2813 - binary_accuracy: 1.0000
    Epoch 3702/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2812 - binary_accuracy: 1.0000
    Epoch 3703/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2812 - binary_accuracy: 1.0000
    Epoch 3704/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2812 - binary_accuracy: 1.0000
    Epoch 3705/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2811 - binary_accuracy: 1.0000
    Epoch 3706/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2811 - binary_accuracy: 1.0000
    Epoch 3707/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2810 - binary_accuracy: 1.0000
    Epoch 3708/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2810 - binary_accuracy: 1.0000
    Epoch 3709/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2809 - binary_accuracy: 1.0000
    Epoch 3710/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2809 - binary_accuracy: 1.0000
    Epoch 3711/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2809 - binary_accuracy: 1.0000
    Epoch 3712/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2808 - binary_accuracy: 1.0000
    Epoch 3713/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2808 - binary_accuracy: 1.0000
    Epoch 3714/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2807 - binary_accuracy: 1.0000
    Epoch 3715/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2807 - binary_accuracy: 1.0000
    Epoch 3716/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2806 - binary_accuracy: 1.0000
    Epoch 3717/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2806 - binary_accuracy: 1.0000
    Epoch 3718/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2806 - binary_accuracy: 1.0000
    Epoch 3719/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2805 - binary_accuracy: 1.0000
    Epoch 3720/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2805 - binary_accuracy: 1.0000
    Epoch 3721/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2804 - binary_accuracy: 1.0000
    Epoch 3722/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2804 - binary_accuracy: 1.0000
    Epoch 3723/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2803 - binary_accuracy: 1.0000
    Epoch 3724/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2803 - binary_accuracy: 1.0000
    Epoch 3725/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2803 - binary_accuracy: 1.0000
    Epoch 3726/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2802 - binary_accuracy: 1.0000
    Epoch 3727/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2802 - binary_accuracy: 1.0000
    Epoch 3728/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2801 - binary_accuracy: 1.0000
    Epoch 3729/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2801 - binary_accuracy: 1.0000
    Epoch 3730/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2800 - binary_accuracy: 1.0000
    Epoch 3731/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2800 - binary_accuracy: 1.0000
    Epoch 3732/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2800 - binary_accuracy: 1.0000
    Epoch 3733/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2799 - binary_accuracy: 1.0000
    Epoch 3734/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2799 - binary_accuracy: 1.0000
    Epoch 3735/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2798 - binary_accuracy: 1.0000
    Epoch 3736/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2798 - binary_accuracy: 1.0000
    Epoch 3737/7000
    1/1 [==============================] - 0s 18ms/step - loss: 0.2798 - binary_accuracy: 1.0000
    Epoch 3738/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2797 - binary_accuracy: 1.0000
    Epoch 3739/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2797 - binary_accuracy: 1.0000
    Epoch 3740/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2796 - binary_accuracy: 1.0000
    Epoch 3741/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2796 - binary_accuracy: 1.0000
    Epoch 3742/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2795 - binary_accuracy: 1.0000
    Epoch 3743/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2795 - binary_accuracy: 1.0000
    Epoch 3744/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2795 - binary_accuracy: 1.0000
    Epoch 3745/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2794 - binary_accuracy: 1.0000
    Epoch 3746/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2794 - binary_accuracy: 1.0000
    Epoch 3747/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.2793 - binary_accuracy: 1.0000
    Epoch 3748/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2793 - binary_accuracy: 1.0000
    Epoch 3749/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2792 - binary_accuracy: 1.0000
    Epoch 3750/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2792 - binary_accuracy: 1.0000
    Epoch 3751/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2792 - binary_accuracy: 1.0000
    Epoch 3752/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2791 - binary_accuracy: 1.0000
    Epoch 3753/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2791 - binary_accuracy: 1.0000
    Epoch 3754/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2790 - binary_accuracy: 1.0000
    Epoch 3755/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2790 - binary_accuracy: 1.0000
    Epoch 3756/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2790 - binary_accuracy: 1.0000
    Epoch 3757/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.2789 - binary_accuracy: 1.0000
    Epoch 3758/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2789 - binary_accuracy: 1.0000
    Epoch 3759/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2788 - binary_accuracy: 1.0000
    Epoch 3760/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2788 - binary_accuracy: 1.0000
    Epoch 3761/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2787 - binary_accuracy: 1.0000
    Epoch 3762/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2787 - binary_accuracy: 1.0000
    Epoch 3763/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2787 - binary_accuracy: 1.0000
    Epoch 3764/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2786 - binary_accuracy: 1.0000
    Epoch 3765/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2786 - binary_accuracy: 1.0000
    Epoch 3766/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2785 - binary_accuracy: 1.0000
    Epoch 3767/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.2785 - binary_accuracy: 1.0000
    Epoch 3768/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2785 - binary_accuracy: 1.0000
    Epoch 3769/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2784 - binary_accuracy: 1.0000
    Epoch 3770/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2784 - binary_accuracy: 1.0000
    Epoch 3771/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2783 - binary_accuracy: 1.0000
    Epoch 3772/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2783 - binary_accuracy: 1.0000
    Epoch 3773/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2782 - binary_accuracy: 1.0000
    Epoch 3774/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2782 - binary_accuracy: 1.0000
    Epoch 3775/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2782 - binary_accuracy: 1.0000
    Epoch 3776/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2781 - binary_accuracy: 1.0000
    Epoch 3777/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2781 - binary_accuracy: 1.0000
    Epoch 3778/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2780 - binary_accuracy: 1.0000
    Epoch 3779/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2780 - binary_accuracy: 1.0000
    Epoch 3780/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2779 - binary_accuracy: 1.0000
    Epoch 3781/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2779 - binary_accuracy: 1.0000
    Epoch 3782/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2779 - binary_accuracy: 1.0000
    Epoch 3783/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2778 - binary_accuracy: 1.0000
    Epoch 3784/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2778 - binary_accuracy: 1.0000
    Epoch 3785/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2777 - binary_accuracy: 1.0000
    Epoch 3786/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2777 - binary_accuracy: 1.0000
    Epoch 3787/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2777 - binary_accuracy: 1.0000
    Epoch 3788/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2776 - binary_accuracy: 1.0000
    Epoch 3789/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2776 - binary_accuracy: 1.0000
    Epoch 3790/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2775 - binary_accuracy: 1.0000
    Epoch 3791/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2775 - binary_accuracy: 1.0000
    Epoch 3792/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2775 - binary_accuracy: 1.0000
    Epoch 3793/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2774 - binary_accuracy: 1.0000
    Epoch 3794/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2774 - binary_accuracy: 1.0000
    Epoch 3795/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2773 - binary_accuracy: 1.0000
    Epoch 3796/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2773 - binary_accuracy: 1.0000
    Epoch 3797/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2772 - binary_accuracy: 1.0000
    Epoch 3798/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2772 - binary_accuracy: 1.0000
    Epoch 3799/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2772 - binary_accuracy: 1.0000
    Epoch 3800/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2771 - binary_accuracy: 1.0000
    Epoch 3801/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2771 - binary_accuracy: 1.0000
    Epoch 3802/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2770 - binary_accuracy: 1.0000
    Epoch 3803/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2770 - binary_accuracy: 1.0000
    Epoch 3804/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2770 - binary_accuracy: 1.0000
    Epoch 3805/7000
    1/1 [==============================] - 0s 21ms/step - loss: 0.2769 - binary_accuracy: 1.0000
    Epoch 3806/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2769 - binary_accuracy: 1.0000
    Epoch 3807/7000
    1/1 [==============================] - 0s 49ms/step - loss: 0.2768 - binary_accuracy: 1.0000
    Epoch 3808/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2768 - binary_accuracy: 1.0000
    Epoch 3809/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2767 - binary_accuracy: 1.0000
    Epoch 3810/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2767 - binary_accuracy: 1.0000
    Epoch 3811/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2767 - binary_accuracy: 1.0000
    Epoch 3812/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2766 - binary_accuracy: 1.0000
    Epoch 3813/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2766 - binary_accuracy: 1.0000
    Epoch 3814/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2765 - binary_accuracy: 1.0000
    Epoch 3815/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2765 - binary_accuracy: 1.0000
    Epoch 3816/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2765 - binary_accuracy: 1.0000
    Epoch 3817/7000
    1/1 [==============================] - 0s 56ms/step - loss: 0.2764 - binary_accuracy: 1.0000
    Epoch 3818/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2764 - binary_accuracy: 1.0000
    Epoch 3819/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2763 - binary_accuracy: 1.0000
    Epoch 3820/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2763 - binary_accuracy: 1.0000
    Epoch 3821/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2763 - binary_accuracy: 1.0000
    Epoch 3822/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2762 - binary_accuracy: 1.0000
    Epoch 3823/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2762 - binary_accuracy: 1.0000
    Epoch 3824/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2761 - binary_accuracy: 1.0000
    Epoch 3825/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2761 - binary_accuracy: 1.0000
    Epoch 3826/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2760 - binary_accuracy: 1.0000
    Epoch 3827/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2760 - binary_accuracy: 1.0000
    Epoch 3828/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2760 - binary_accuracy: 1.0000
    Epoch 3829/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2759 - binary_accuracy: 1.0000
    Epoch 3830/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2759 - binary_accuracy: 1.0000
    Epoch 3831/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2758 - binary_accuracy: 1.0000
    Epoch 3832/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2758 - binary_accuracy: 1.0000
    Epoch 3833/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2758 - binary_accuracy: 1.0000
    Epoch 3834/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2757 - binary_accuracy: 1.0000
    Epoch 3835/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2757 - binary_accuracy: 1.0000
    Epoch 3836/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2756 - binary_accuracy: 1.0000
    Epoch 3837/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2756 - binary_accuracy: 1.0000
    Epoch 3838/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2756 - binary_accuracy: 1.0000
    Epoch 3839/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2755 - binary_accuracy: 1.0000
    Epoch 3840/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2755 - binary_accuracy: 1.0000
    Epoch 3841/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2754 - binary_accuracy: 1.0000
    Epoch 3842/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2754 - binary_accuracy: 1.0000
    Epoch 3843/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2754 - binary_accuracy: 1.0000
    Epoch 3844/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2753 - binary_accuracy: 1.0000
    Epoch 3845/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2753 - binary_accuracy: 1.0000
    Epoch 3846/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.2752 - binary_accuracy: 1.0000
    Epoch 3847/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2752 - binary_accuracy: 1.0000
    Epoch 3848/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2752 - binary_accuracy: 1.0000
    Epoch 3849/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2751 - binary_accuracy: 1.0000
    Epoch 3850/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2751 - binary_accuracy: 1.0000
    Epoch 3851/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2750 - binary_accuracy: 1.0000
    Epoch 3852/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2750 - binary_accuracy: 1.0000
    Epoch 3853/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2749 - binary_accuracy: 1.0000
    Epoch 3854/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2749 - binary_accuracy: 1.0000
    Epoch 3855/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2749 - binary_accuracy: 1.0000
    Epoch 3856/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.2748 - binary_accuracy: 1.0000
    Epoch 3857/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2748 - binary_accuracy: 1.0000
    Epoch 3858/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2747 - binary_accuracy: 1.0000
    Epoch 3859/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2747 - binary_accuracy: 1.0000
    Epoch 3860/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2747 - binary_accuracy: 1.0000
    Epoch 3861/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2746 - binary_accuracy: 1.0000
    Epoch 3862/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2746 - binary_accuracy: 1.0000
    Epoch 3863/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2745 - binary_accuracy: 1.0000
    Epoch 3864/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2745 - binary_accuracy: 1.0000
    Epoch 3865/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2745 - binary_accuracy: 1.0000
    Epoch 3866/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2744 - binary_accuracy: 1.0000
    Epoch 3867/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2744 - binary_accuracy: 1.0000
    Epoch 3868/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2743 - binary_accuracy: 1.0000
    Epoch 3869/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2743 - binary_accuracy: 1.0000
    Epoch 3870/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2743 - binary_accuracy: 1.0000
    Epoch 3871/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2742 - binary_accuracy: 1.0000
    Epoch 3872/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2742 - binary_accuracy: 1.0000
    Epoch 3873/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2741 - binary_accuracy: 1.0000
    Epoch 3874/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2741 - binary_accuracy: 1.0000
    Epoch 3875/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2741 - binary_accuracy: 1.0000
    Epoch 3876/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2740 - binary_accuracy: 1.0000
    Epoch 3877/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2740 - binary_accuracy: 1.0000
    Epoch 3878/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2739 - binary_accuracy: 1.0000
    Epoch 3879/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2739 - binary_accuracy: 1.0000
    Epoch 3880/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2739 - binary_accuracy: 1.0000
    Epoch 3881/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2738 - binary_accuracy: 1.0000
    Epoch 3882/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2738 - binary_accuracy: 1.0000
    Epoch 3883/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2737 - binary_accuracy: 1.0000
    Epoch 3884/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2737 - binary_accuracy: 1.0000
    Epoch 3885/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2737 - binary_accuracy: 1.0000
    Epoch 3886/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2736 - binary_accuracy: 1.0000
    Epoch 3887/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2736 - binary_accuracy: 1.0000
    Epoch 3888/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2735 - binary_accuracy: 1.0000
    Epoch 3889/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2735 - binary_accuracy: 1.0000
    Epoch 3890/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2735 - binary_accuracy: 1.0000
    Epoch 3891/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2734 - binary_accuracy: 1.0000
    Epoch 3892/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2734 - binary_accuracy: 1.0000
    Epoch 3893/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2733 - binary_accuracy: 1.0000
    Epoch 3894/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2733 - binary_accuracy: 1.0000
    Epoch 3895/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2733 - binary_accuracy: 1.0000
    Epoch 3896/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2732 - binary_accuracy: 1.0000
    Epoch 3897/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2732 - binary_accuracy: 1.0000
    Epoch 3898/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2731 - binary_accuracy: 1.0000
    Epoch 3899/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2731 - binary_accuracy: 1.0000
    Epoch 3900/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2731 - binary_accuracy: 1.0000
    Epoch 3901/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2730 - binary_accuracy: 1.0000
    Epoch 3902/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2730 - binary_accuracy: 1.0000
    Epoch 3903/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2729 - binary_accuracy: 1.0000
    Epoch 3904/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2729 - binary_accuracy: 1.0000
    Epoch 3905/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2729 - binary_accuracy: 1.0000
    Epoch 3906/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2728 - binary_accuracy: 1.0000
    Epoch 3907/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2728 - binary_accuracy: 1.0000
    Epoch 3908/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2727 - binary_accuracy: 1.0000
    Epoch 3909/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2727 - binary_accuracy: 1.0000
    Epoch 3910/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2727 - binary_accuracy: 1.0000
    Epoch 3911/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2726 - binary_accuracy: 1.0000
    Epoch 3912/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2726 - binary_accuracy: 1.0000
    Epoch 3913/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2725 - binary_accuracy: 1.0000
    Epoch 3914/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2725 - binary_accuracy: 1.0000
    Epoch 3915/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2725 - binary_accuracy: 1.0000
    Epoch 3916/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2724 - binary_accuracy: 1.0000
    Epoch 3917/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2724 - binary_accuracy: 1.0000
    Epoch 3918/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2723 - binary_accuracy: 1.0000
    Epoch 3919/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2723 - binary_accuracy: 1.0000
    Epoch 3920/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2723 - binary_accuracy: 1.0000
    Epoch 3921/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2722 - binary_accuracy: 1.0000
    Epoch 3922/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2722 - binary_accuracy: 1.0000
    Epoch 3923/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2721 - binary_accuracy: 1.0000
    Epoch 3924/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2721 - binary_accuracy: 1.0000
    Epoch 3925/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2721 - binary_accuracy: 1.0000
    Epoch 3926/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2720 - binary_accuracy: 1.0000
    Epoch 3927/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2720 - binary_accuracy: 1.0000
    Epoch 3928/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2719 - binary_accuracy: 1.0000
    Epoch 3929/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2719 - binary_accuracy: 1.0000
    Epoch 3930/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2719 - binary_accuracy: 1.0000
    Epoch 3931/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2718 - binary_accuracy: 1.0000
    Epoch 3932/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2718 - binary_accuracy: 1.0000
    Epoch 3933/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2717 - binary_accuracy: 1.0000
    Epoch 3934/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.2717 - binary_accuracy: 1.0000
    Epoch 3935/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2717 - binary_accuracy: 1.0000
    Epoch 3936/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2716 - binary_accuracy: 1.0000
    Epoch 3937/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2716 - binary_accuracy: 1.0000
    Epoch 3938/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2715 - binary_accuracy: 1.0000
    Epoch 3939/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2715 - binary_accuracy: 1.0000
    Epoch 3940/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2715 - binary_accuracy: 1.0000
    Epoch 3941/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2714 - binary_accuracy: 1.0000
    Epoch 3942/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2714 - binary_accuracy: 1.0000
    Epoch 3943/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2713 - binary_accuracy: 1.0000
    Epoch 3944/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2713 - binary_accuracy: 1.0000
    Epoch 3945/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2713 - binary_accuracy: 1.0000
    Epoch 3946/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2712 - binary_accuracy: 1.0000
    Epoch 3947/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2712 - binary_accuracy: 1.0000
    Epoch 3948/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2711 - binary_accuracy: 1.0000
    Epoch 3949/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2711 - binary_accuracy: 1.0000
    Epoch 3950/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2711 - binary_accuracy: 1.0000
    Epoch 3951/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2710 - binary_accuracy: 1.0000
    Epoch 3952/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2710 - binary_accuracy: 1.0000
    Epoch 3953/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2709 - binary_accuracy: 1.0000
    Epoch 3954/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2709 - binary_accuracy: 1.0000
    Epoch 3955/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2709 - binary_accuracy: 1.0000
    Epoch 3956/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2708 - binary_accuracy: 1.0000
    Epoch 3957/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2708 - binary_accuracy: 1.0000
    Epoch 3958/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2708 - binary_accuracy: 1.0000
    Epoch 3959/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2707 - binary_accuracy: 1.0000
    Epoch 3960/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2707 - binary_accuracy: 1.0000
    Epoch 3961/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2706 - binary_accuracy: 1.0000
    Epoch 3962/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2706 - binary_accuracy: 1.0000
    Epoch 3963/7000
    1/1 [==============================] - 0s 41ms/step - loss: 0.2706 - binary_accuracy: 1.0000
    Epoch 3964/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2705 - binary_accuracy: 1.0000
    Epoch 3965/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2705 - binary_accuracy: 1.0000
    Epoch 3966/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2704 - binary_accuracy: 1.0000
    Epoch 3967/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2704 - binary_accuracy: 1.0000
    Epoch 3968/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2704 - binary_accuracy: 1.0000
    Epoch 3969/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2703 - binary_accuracy: 1.0000
    Epoch 3970/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2703 - binary_accuracy: 1.0000
    Epoch 3971/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2702 - binary_accuracy: 1.0000
    Epoch 3972/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2702 - binary_accuracy: 1.0000
    Epoch 3973/7000
    1/1 [==============================] - 0s 49ms/step - loss: 0.2702 - binary_accuracy: 1.0000
    Epoch 3974/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2701 - binary_accuracy: 1.0000
    Epoch 3975/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2701 - binary_accuracy: 1.0000
    Epoch 3976/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2700 - binary_accuracy: 1.0000
    Epoch 3977/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2700 - binary_accuracy: 1.0000
    Epoch 3978/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2700 - binary_accuracy: 1.0000
    Epoch 3979/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2699 - binary_accuracy: 1.0000
    Epoch 3980/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2699 - binary_accuracy: 1.0000
    Epoch 3981/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2699 - binary_accuracy: 1.0000
    Epoch 3982/7000
    1/1 [==============================] - 0s 50ms/step - loss: 0.2698 - binary_accuracy: 1.0000
    Epoch 3983/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2698 - binary_accuracy: 1.0000
    Epoch 3984/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2697 - binary_accuracy: 1.0000
    Epoch 3985/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2697 - binary_accuracy: 1.0000
    Epoch 3986/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2697 - binary_accuracy: 1.0000
    Epoch 3987/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2696 - binary_accuracy: 1.0000
    Epoch 3988/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2696 - binary_accuracy: 1.0000
    Epoch 3989/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2695 - binary_accuracy: 1.0000
    Epoch 3990/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2695 - binary_accuracy: 1.0000
    Epoch 3991/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2695 - binary_accuracy: 1.0000
    Epoch 3992/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.2694 - binary_accuracy: 1.0000
    Epoch 3993/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2694 - binary_accuracy: 1.0000
    Epoch 3994/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2693 - binary_accuracy: 1.0000
    Epoch 3995/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2693 - binary_accuracy: 1.0000
    Epoch 3996/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2693 - binary_accuracy: 1.0000
    Epoch 3997/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2692 - binary_accuracy: 1.0000
    Epoch 3998/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2692 - binary_accuracy: 1.0000
    Epoch 3999/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2692 - binary_accuracy: 1.0000
    Epoch 4000/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2691 - binary_accuracy: 1.0000
    Epoch 4001/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2691 - binary_accuracy: 1.0000
    Epoch 4002/7000
    1/1 [==============================] - 0s 15ms/step - loss: 0.2690 - binary_accuracy: 1.0000
    Epoch 4003/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2690 - binary_accuracy: 1.0000
    Epoch 4004/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2690 - binary_accuracy: 1.0000
    Epoch 4005/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2689 - binary_accuracy: 1.0000
    Epoch 4006/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2689 - binary_accuracy: 1.0000
    Epoch 4007/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2688 - binary_accuracy: 1.0000
    Epoch 4008/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2688 - binary_accuracy: 1.0000
    Epoch 4009/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2688 - binary_accuracy: 1.0000
    Epoch 4010/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2687 - binary_accuracy: 1.0000
    Epoch 4011/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2687 - binary_accuracy: 1.0000
    Epoch 4012/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2686 - binary_accuracy: 1.0000
    Epoch 4013/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2686 - binary_accuracy: 1.0000
    Epoch 4014/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2686 - binary_accuracy: 1.0000
    Epoch 4015/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2685 - binary_accuracy: 1.0000
    Epoch 4016/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2685 - binary_accuracy: 1.0000
    Epoch 4017/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2685 - binary_accuracy: 1.0000
    Epoch 4018/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2684 - binary_accuracy: 1.0000
    Epoch 4019/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2684 - binary_accuracy: 1.0000
    Epoch 4020/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2683 - binary_accuracy: 1.0000
    Epoch 4021/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.2683 - binary_accuracy: 1.0000
    Epoch 4022/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2683 - binary_accuracy: 1.0000
    Epoch 4023/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2682 - binary_accuracy: 1.0000
    Epoch 4024/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2682 - binary_accuracy: 1.0000
    Epoch 4025/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2681 - binary_accuracy: 1.0000
    Epoch 4026/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2681 - binary_accuracy: 1.0000
    Epoch 4027/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2681 - binary_accuracy: 1.0000
    Epoch 4028/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2680 - binary_accuracy: 1.0000
    Epoch 4029/7000
    1/1 [==============================] - 0s 43ms/step - loss: 0.2680 - binary_accuracy: 1.0000
    Epoch 4030/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2680 - binary_accuracy: 1.0000
    Epoch 4031/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2679 - binary_accuracy: 1.0000
    Epoch 4032/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2679 - binary_accuracy: 1.0000
    Epoch 4033/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2678 - binary_accuracy: 1.0000
    Epoch 4034/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2678 - binary_accuracy: 1.0000
    Epoch 4035/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2678 - binary_accuracy: 1.0000
    Epoch 4036/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2677 - binary_accuracy: 1.0000
    Epoch 4037/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2677 - binary_accuracy: 1.0000
    Epoch 4038/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.2676 - binary_accuracy: 1.0000
    Epoch 4039/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2676 - binary_accuracy: 1.0000
    Epoch 4040/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2676 - binary_accuracy: 1.0000
    Epoch 4041/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2675 - binary_accuracy: 1.0000
    Epoch 4042/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2675 - binary_accuracy: 1.0000
    Epoch 4043/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2675 - binary_accuracy: 1.0000
    Epoch 4044/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2674 - binary_accuracy: 1.0000
    Epoch 4045/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2674 - binary_accuracy: 1.0000
    Epoch 4046/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2673 - binary_accuracy: 1.0000
    Epoch 4047/7000
    1/1 [==============================] - 0s 51ms/step - loss: 0.2673 - binary_accuracy: 1.0000
    Epoch 4048/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2673 - binary_accuracy: 1.0000
    Epoch 4049/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2672 - binary_accuracy: 1.0000
    Epoch 4050/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2672 - binary_accuracy: 1.0000
    Epoch 4051/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2671 - binary_accuracy: 1.0000
    Epoch 4052/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2671 - binary_accuracy: 1.0000
    Epoch 4053/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2671 - binary_accuracy: 1.0000
    Epoch 4054/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2670 - binary_accuracy: 1.0000
    Epoch 4055/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2670 - binary_accuracy: 1.0000
    Epoch 4056/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2670 - binary_accuracy: 1.0000
    Epoch 4057/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2669 - binary_accuracy: 1.0000
    Epoch 4058/7000
    1/1 [==============================] - 0s 96ms/step - loss: 0.2669 - binary_accuracy: 1.0000
    Epoch 4059/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2668 - binary_accuracy: 1.0000
    Epoch 4060/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2668 - binary_accuracy: 1.0000
    Epoch 4061/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2668 - binary_accuracy: 1.0000
    Epoch 4062/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2667 - binary_accuracy: 1.0000
    Epoch 4063/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2667 - binary_accuracy: 1.0000
    Epoch 4064/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2667 - binary_accuracy: 1.0000
    Epoch 4065/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2666 - binary_accuracy: 1.0000
    Epoch 4066/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2666 - binary_accuracy: 1.0000
    Epoch 4067/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2665 - binary_accuracy: 1.0000
    Epoch 4068/7000
    1/1 [==============================] - 0s 16ms/step - loss: 0.2665 - binary_accuracy: 1.0000
    Epoch 4069/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2665 - binary_accuracy: 1.0000
    Epoch 4070/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2664 - binary_accuracy: 1.0000
    Epoch 4071/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2664 - binary_accuracy: 1.0000
    Epoch 4072/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2663 - binary_accuracy: 1.0000
    Epoch 4073/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2663 - binary_accuracy: 1.0000
    Epoch 4074/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2663 - binary_accuracy: 1.0000
    Epoch 4075/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2662 - binary_accuracy: 1.0000
    Epoch 4076/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2662 - binary_accuracy: 1.0000
    Epoch 4077/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2662 - binary_accuracy: 1.0000
    Epoch 4078/7000
    1/1 [==============================] - 0s 39ms/step - loss: 0.2661 - binary_accuracy: 1.0000
    Epoch 4079/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2661 - binary_accuracy: 1.0000
    Epoch 4080/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2660 - binary_accuracy: 1.0000
    Epoch 4081/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2660 - binary_accuracy: 1.0000
    Epoch 4082/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2660 - binary_accuracy: 1.0000
    Epoch 4083/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2659 - binary_accuracy: 1.0000
    Epoch 4084/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2659 - binary_accuracy: 1.0000
    Epoch 4085/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2659 - binary_accuracy: 1.0000
    Epoch 4086/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2658 - binary_accuracy: 1.0000
    Epoch 4087/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2658 - binary_accuracy: 1.0000
    Epoch 4088/7000
    1/1 [==============================] - 0s 33ms/step - loss: 0.2657 - binary_accuracy: 1.0000
    Epoch 4089/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2657 - binary_accuracy: 1.0000
    Epoch 4090/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2657 - binary_accuracy: 1.0000
    Epoch 4091/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2656 - binary_accuracy: 1.0000
    Epoch 4092/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2656 - binary_accuracy: 1.0000
    Epoch 4093/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2656 - binary_accuracy: 1.0000
    Epoch 4094/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2655 - binary_accuracy: 1.0000
    Epoch 4095/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2655 - binary_accuracy: 1.0000
    Epoch 4096/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2654 - binary_accuracy: 1.0000
    Epoch 4097/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2654 - binary_accuracy: 1.0000
    Epoch 4098/7000
    1/1 [==============================] - 0s 40ms/step - loss: 0.2654 - binary_accuracy: 1.0000
    Epoch 4099/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2653 - binary_accuracy: 1.0000
    Epoch 4100/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2653 - binary_accuracy: 1.0000
    Epoch 4101/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2653 - binary_accuracy: 1.0000
    Epoch 4102/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2652 - binary_accuracy: 1.0000
    Epoch 4103/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2652 - binary_accuracy: 1.0000
    Epoch 4104/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2651 - binary_accuracy: 1.0000
    Epoch 4105/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2651 - binary_accuracy: 1.0000
    Epoch 4106/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2651 - binary_accuracy: 1.0000
    Epoch 4107/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2650 - binary_accuracy: 1.0000
    Epoch 4108/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2650 - binary_accuracy: 1.0000
    Epoch 4109/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2650 - binary_accuracy: 1.0000
    Epoch 4110/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2649 - binary_accuracy: 1.0000
    Epoch 4111/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2649 - binary_accuracy: 1.0000
    Epoch 4112/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2648 - binary_accuracy: 1.0000
    Epoch 4113/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2648 - binary_accuracy: 1.0000
    Epoch 4114/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2648 - binary_accuracy: 1.0000
    Epoch 4115/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2647 - binary_accuracy: 1.0000
    Epoch 4116/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2647 - binary_accuracy: 1.0000
    Epoch 4117/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2647 - binary_accuracy: 1.0000
    Epoch 4118/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2646 - binary_accuracy: 1.0000
    Epoch 4119/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2646 - binary_accuracy: 1.0000
    Epoch 4120/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2645 - binary_accuracy: 1.0000
    Epoch 4121/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2645 - binary_accuracy: 1.0000
    Epoch 4122/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2645 - binary_accuracy: 1.0000
    Epoch 4123/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2644 - binary_accuracy: 1.0000
    Epoch 4124/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2644 - binary_accuracy: 1.0000
    Epoch 4125/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2644 - binary_accuracy: 1.0000
    Epoch 4126/7000
    1/1 [==============================] - 0s 46ms/step - loss: 0.2643 - binary_accuracy: 1.0000
    Epoch 4127/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2643 - binary_accuracy: 1.0000
    Epoch 4128/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2642 - binary_accuracy: 1.0000
    Epoch 4129/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2642 - binary_accuracy: 1.0000
    Epoch 4130/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2642 - binary_accuracy: 1.0000
    Epoch 4131/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2641 - binary_accuracy: 1.0000
    Epoch 4132/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2641 - binary_accuracy: 1.0000
    Epoch 4133/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2641 - binary_accuracy: 1.0000
    Epoch 4134/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2640 - binary_accuracy: 1.0000
    Epoch 4135/7000
    1/1 [==============================] - 0s 20ms/step - loss: 0.2640 - binary_accuracy: 1.0000
    Epoch 4136/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2639 - binary_accuracy: 1.0000
    Epoch 4137/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2639 - binary_accuracy: 1.0000
    Epoch 4138/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2639 - binary_accuracy: 1.0000
    Epoch 4139/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2638 - binary_accuracy: 1.0000
    Epoch 4140/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2638 - binary_accuracy: 1.0000
    Epoch 4141/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2638 - binary_accuracy: 1.0000
    Epoch 4142/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2637 - binary_accuracy: 1.0000
    Epoch 4143/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2637 - binary_accuracy: 1.0000
    Epoch 4144/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2636 - binary_accuracy: 1.0000
    Epoch 4145/7000
    1/1 [==============================] - 0s 53ms/step - loss: 0.2636 - binary_accuracy: 1.0000
    Epoch 4146/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2636 - binary_accuracy: 1.0000
    Epoch 4147/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2635 - binary_accuracy: 1.0000
    Epoch 4148/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2635 - binary_accuracy: 1.0000
    Epoch 4149/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2635 - binary_accuracy: 1.0000
    Epoch 4150/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2634 - binary_accuracy: 1.0000
    Epoch 4151/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2634 - binary_accuracy: 1.0000
    Epoch 4152/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2634 - binary_accuracy: 1.0000
    Epoch 4153/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2633 - binary_accuracy: 1.0000
    Epoch 4154/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2633 - binary_accuracy: 1.0000
    Epoch 4155/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2632 - binary_accuracy: 1.0000
    Epoch 4156/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2632 - binary_accuracy: 1.0000
    Epoch 4157/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2632 - binary_accuracy: 1.0000
    Epoch 4158/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2631 - binary_accuracy: 1.0000
    Epoch 4159/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2631 - binary_accuracy: 1.0000
    Epoch 4160/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2631 - binary_accuracy: 1.0000
    Epoch 4161/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2630 - binary_accuracy: 1.0000
    Epoch 4162/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2630 - binary_accuracy: 1.0000
    Epoch 4163/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2629 - binary_accuracy: 1.0000
    Epoch 4164/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2629 - binary_accuracy: 1.0000
    Epoch 4165/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2629 - binary_accuracy: 1.0000
    Epoch 4166/7000
    1/1 [==============================] - 0s 71ms/step - loss: 0.2628 - binary_accuracy: 1.0000
    Epoch 4167/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2628 - binary_accuracy: 1.0000
    Epoch 4168/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2628 - binary_accuracy: 1.0000
    Epoch 4169/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2627 - binary_accuracy: 1.0000
    Epoch 4170/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2627 - binary_accuracy: 1.0000
    Epoch 4171/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2627 - binary_accuracy: 1.0000
    Epoch 4172/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2626 - binary_accuracy: 1.0000
    Epoch 4173/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2626 - binary_accuracy: 1.0000
    Epoch 4174/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2625 - binary_accuracy: 1.0000
    Epoch 4175/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2625 - binary_accuracy: 1.0000
    Epoch 4176/7000
    1/1 [==============================] - 0s 56ms/step - loss: 0.2625 - binary_accuracy: 1.0000
    Epoch 4177/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2624 - binary_accuracy: 1.0000
    Epoch 4178/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2624 - binary_accuracy: 1.0000
    Epoch 4179/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2624 - binary_accuracy: 1.0000
    Epoch 4180/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2623 - binary_accuracy: 1.0000
    Epoch 4181/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2623 - binary_accuracy: 1.0000
    Epoch 4182/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2622 - binary_accuracy: 1.0000
    Epoch 4183/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2622 - binary_accuracy: 1.0000
    Epoch 4184/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2622 - binary_accuracy: 1.0000
    Epoch 4185/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2621 - binary_accuracy: 1.0000
    Epoch 4186/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2621 - binary_accuracy: 1.0000
    Epoch 4187/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2621 - binary_accuracy: 1.0000
    Epoch 4188/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2620 - binary_accuracy: 1.0000
    Epoch 4189/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2620 - binary_accuracy: 1.0000
    Epoch 4190/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2620 - binary_accuracy: 1.0000
    Epoch 4191/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2619 - binary_accuracy: 1.0000
    Epoch 4192/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2619 - binary_accuracy: 1.0000
    Epoch 4193/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2618 - binary_accuracy: 1.0000
    Epoch 4194/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2618 - binary_accuracy: 1.0000
    Epoch 4195/7000
    1/1 [==============================] - 0s 34ms/step - loss: 0.2618 - binary_accuracy: 1.0000
    Epoch 4196/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2617 - binary_accuracy: 1.0000
    Epoch 4197/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2617 - binary_accuracy: 1.0000
    Epoch 4198/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2617 - binary_accuracy: 1.0000
    Epoch 4199/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2616 - binary_accuracy: 1.0000
    Epoch 4200/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2616 - binary_accuracy: 1.0000
    Epoch 4201/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2616 - binary_accuracy: 1.0000
    Epoch 4202/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2615 - binary_accuracy: 1.0000
    Epoch 4203/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2615 - binary_accuracy: 1.0000
    Epoch 4204/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2614 - binary_accuracy: 1.0000
    Epoch 4205/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2614 - binary_accuracy: 1.0000
    Epoch 4206/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.2614 - binary_accuracy: 1.0000
    Epoch 4207/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2613 - binary_accuracy: 1.0000
    Epoch 4208/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2613 - binary_accuracy: 1.0000
    Epoch 4209/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2613 - binary_accuracy: 1.0000
    Epoch 4210/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2612 - binary_accuracy: 1.0000
    Epoch 4211/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2612 - binary_accuracy: 1.0000
    Epoch 4212/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2612 - binary_accuracy: 1.0000
    Epoch 4213/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2611 - binary_accuracy: 1.0000
    Epoch 4214/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2611 - binary_accuracy: 1.0000
    Epoch 4215/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2610 - binary_accuracy: 1.0000
    Epoch 4216/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.2610 - binary_accuracy: 1.0000
    Epoch 4217/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2610 - binary_accuracy: 1.0000
    Epoch 4218/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2609 - binary_accuracy: 1.0000
    Epoch 4219/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2609 - binary_accuracy: 1.0000
    Epoch 4220/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2609 - binary_accuracy: 1.0000
    Epoch 4221/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2608 - binary_accuracy: 1.0000
    Epoch 4222/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2608 - binary_accuracy: 1.0000
    Epoch 4223/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2608 - binary_accuracy: 1.0000
    Epoch 4224/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2607 - binary_accuracy: 1.0000
    Epoch 4225/7000
    1/1 [==============================] - 0s 31ms/step - loss: 0.2607 - binary_accuracy: 1.0000
    Epoch 4226/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2606 - binary_accuracy: 1.0000
    Epoch 4227/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2606 - binary_accuracy: 1.0000
    Epoch 4228/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2606 - binary_accuracy: 1.0000
    Epoch 4229/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2605 - binary_accuracy: 1.0000
    Epoch 4230/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2605 - binary_accuracy: 1.0000
    Epoch 4231/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2605 - binary_accuracy: 1.0000
    Epoch 4232/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2604 - binary_accuracy: 1.0000
    Epoch 4233/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2604 - binary_accuracy: 1.0000
    Epoch 4234/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2604 - binary_accuracy: 1.0000
    Epoch 4235/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2603 - binary_accuracy: 1.0000
    Epoch 4236/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2603 - binary_accuracy: 1.0000
    Epoch 4237/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2602 - binary_accuracy: 1.0000
    Epoch 4238/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2602 - binary_accuracy: 1.0000
    Epoch 4239/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2602 - binary_accuracy: 1.0000
    Epoch 4240/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2601 - binary_accuracy: 1.0000
    Epoch 4241/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2601 - binary_accuracy: 1.0000
    Epoch 4242/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2601 - binary_accuracy: 1.0000
    Epoch 4243/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2600 - binary_accuracy: 1.0000
    Epoch 4244/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2600 - binary_accuracy: 1.0000
    Epoch 4245/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2600 - binary_accuracy: 1.0000
    Epoch 4246/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2599 - binary_accuracy: 1.0000
    Epoch 4247/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2599 - binary_accuracy: 1.0000
    Epoch 4248/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2599 - binary_accuracy: 1.0000
    Epoch 4249/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2598 - binary_accuracy: 1.0000
    Epoch 4250/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2598 - binary_accuracy: 1.0000
    Epoch 4251/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2597 - binary_accuracy: 1.0000
    Epoch 4252/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2597 - binary_accuracy: 1.0000
    Epoch 4253/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2597 - binary_accuracy: 1.0000
    Epoch 4254/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2596 - binary_accuracy: 1.0000
    Epoch 4255/7000
    1/1 [==============================] - 0s 28ms/step - loss: 0.2596 - binary_accuracy: 1.0000
    Epoch 4256/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2596 - binary_accuracy: 1.0000
    Epoch 4257/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2595 - binary_accuracy: 1.0000
    Epoch 4258/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2595 - binary_accuracy: 1.0000
    Epoch 4259/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2595 - binary_accuracy: 1.0000
    Epoch 4260/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2594 - binary_accuracy: 1.0000
    Epoch 4261/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2594 - binary_accuracy: 1.0000
    Epoch 4262/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2593 - binary_accuracy: 1.0000
    Epoch 4263/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2593 - binary_accuracy: 1.0000
    Epoch 4264/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2593 - binary_accuracy: 1.0000
    Epoch 4265/7000
    1/1 [==============================] - 0s 50ms/step - loss: 0.2592 - binary_accuracy: 1.0000
    Epoch 4266/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2592 - binary_accuracy: 1.0000
    Epoch 4267/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2592 - binary_accuracy: 1.0000
    Epoch 4268/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2591 - binary_accuracy: 1.0000
    Epoch 4269/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2591 - binary_accuracy: 1.0000
    Epoch 4270/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2591 - binary_accuracy: 1.0000
    Epoch 4271/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2590 - binary_accuracy: 1.0000
    Epoch 4272/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2590 - binary_accuracy: 1.0000
    Epoch 4273/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2590 - binary_accuracy: 1.0000
    Epoch 4274/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2589 - binary_accuracy: 1.0000
    Epoch 4275/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2589 - binary_accuracy: 1.0000
    Epoch 4276/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2588 - binary_accuracy: 1.0000
    Epoch 4277/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2588 - binary_accuracy: 1.0000
    Epoch 4278/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2588 - binary_accuracy: 1.0000
    Epoch 4279/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2587 - binary_accuracy: 1.0000
    Epoch 4280/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2587 - binary_accuracy: 1.0000
    Epoch 4281/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2587 - binary_accuracy: 1.0000
    Epoch 4282/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2586 - binary_accuracy: 1.0000
    Epoch 4283/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2586 - binary_accuracy: 1.0000
    Epoch 4284/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2586 - binary_accuracy: 1.0000
    Epoch 4285/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2585 - binary_accuracy: 1.0000
    Epoch 4286/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2585 - binary_accuracy: 1.0000
    Epoch 4287/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2585 - binary_accuracy: 1.0000
    Epoch 4288/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2584 - binary_accuracy: 1.0000
    Epoch 4289/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2584 - binary_accuracy: 1.0000
    Epoch 4290/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2584 - binary_accuracy: 1.0000
    Epoch 4291/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2583 - binary_accuracy: 1.0000
    Epoch 4292/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2583 - binary_accuracy: 1.0000
    Epoch 4293/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2582 - binary_accuracy: 1.0000
    Epoch 4294/7000
    1/1 [==============================] - 0s 22ms/step - loss: 0.2582 - binary_accuracy: 1.0000
    Epoch 4295/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2582 - binary_accuracy: 1.0000
    Epoch 4296/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2581 - binary_accuracy: 1.0000
    Epoch 4297/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2581 - binary_accuracy: 1.0000
    Epoch 4298/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2581 - binary_accuracy: 1.0000
    Epoch 4299/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2580 - binary_accuracy: 1.0000
    Epoch 4300/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2580 - binary_accuracy: 1.0000
    Epoch 4301/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2580 - binary_accuracy: 1.0000
    Epoch 4302/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2579 - binary_accuracy: 1.0000
    Epoch 4303/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2579 - binary_accuracy: 1.0000
    Epoch 4304/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2579 - binary_accuracy: 1.0000
    Epoch 4305/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2578 - binary_accuracy: 1.0000
    Epoch 4306/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2578 - binary_accuracy: 1.0000
    Epoch 4307/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2577 - binary_accuracy: 1.0000
    Epoch 4308/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2577 - binary_accuracy: 1.0000
    Epoch 4309/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2577 - binary_accuracy: 1.0000
    Epoch 4310/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2576 - binary_accuracy: 1.0000
    Epoch 4311/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2576 - binary_accuracy: 1.0000
    Epoch 4312/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2576 - binary_accuracy: 1.0000
    Epoch 4313/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2575 - binary_accuracy: 1.0000
    Epoch 4314/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2575 - binary_accuracy: 1.0000
    Epoch 4315/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2575 - binary_accuracy: 1.0000
    Epoch 4316/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2574 - binary_accuracy: 1.0000
    Epoch 4317/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2574 - binary_accuracy: 1.0000
    Epoch 4318/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2574 - binary_accuracy: 1.0000
    Epoch 4319/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2573 - binary_accuracy: 1.0000
    Epoch 4320/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2573 - binary_accuracy: 1.0000
    Epoch 4321/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2573 - binary_accuracy: 1.0000
    Epoch 4322/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2572 - binary_accuracy: 1.0000
    Epoch 4323/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2572 - binary_accuracy: 1.0000
    Epoch 4324/7000
    1/1 [==============================] - 0s 37ms/step - loss: 0.2571 - binary_accuracy: 1.0000
    Epoch 4325/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2571 - binary_accuracy: 1.0000
    Epoch 4326/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2571 - binary_accuracy: 1.0000
    Epoch 4327/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2570 - binary_accuracy: 1.0000
    Epoch 4328/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2570 - binary_accuracy: 1.0000
    Epoch 4329/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2570 - binary_accuracy: 1.0000
    Epoch 4330/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2569 - binary_accuracy: 1.0000
    Epoch 4331/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2569 - binary_accuracy: 1.0000
    Epoch 4332/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2569 - binary_accuracy: 1.0000
    Epoch 4333/7000
    1/1 [==============================] - 0s 44ms/step - loss: 0.2568 - binary_accuracy: 1.0000
    Epoch 4334/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2568 - binary_accuracy: 1.0000
    Epoch 4335/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2568 - binary_accuracy: 1.0000
    Epoch 4336/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2567 - binary_accuracy: 1.0000
    Epoch 4337/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2567 - binary_accuracy: 1.0000
    Epoch 4338/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2567 - binary_accuracy: 1.0000
    Epoch 4339/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2566 - binary_accuracy: 1.0000
    Epoch 4340/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2566 - binary_accuracy: 1.0000
    Epoch 4341/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2566 - binary_accuracy: 1.0000
    Epoch 4342/7000
    1/1 [==============================] - 0s 38ms/step - loss: 0.2565 - binary_accuracy: 1.0000
    Epoch 4343/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2565 - binary_accuracy: 1.0000
    Epoch 4344/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2564 - binary_accuracy: 1.0000
    Epoch 4345/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2564 - binary_accuracy: 1.0000
    Epoch 4346/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2564 - binary_accuracy: 1.0000
    Epoch 4347/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2563 - binary_accuracy: 1.0000
    Epoch 4348/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2563 - binary_accuracy: 1.0000
    Epoch 4349/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2563 - binary_accuracy: 1.0000
    Epoch 4350/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2562 - binary_accuracy: 1.0000
    Epoch 4351/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2562 - binary_accuracy: 1.0000
    Epoch 4352/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2562 - binary_accuracy: 1.0000
    Epoch 4353/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2561 - binary_accuracy: 1.0000
    Epoch 4354/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2561 - binary_accuracy: 1.0000
    Epoch 4355/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2561 - binary_accuracy: 1.0000
    Epoch 4356/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2560 - binary_accuracy: 1.0000
    Epoch 4357/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2560 - binary_accuracy: 1.0000
    Epoch 4358/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2560 - binary_accuracy: 1.0000
    Epoch 4359/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2559 - binary_accuracy: 1.0000
    Epoch 4360/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2559 - binary_accuracy: 1.0000
    Epoch 4361/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2559 - binary_accuracy: 1.0000
    Epoch 4362/7000
    1/1 [==============================] - 0s 26ms/step - loss: 0.2558 - binary_accuracy: 1.0000
    Epoch 4363/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2558 - binary_accuracy: 1.0000
    Epoch 4364/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2558 - binary_accuracy: 1.0000
    Epoch 4365/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2557 - binary_accuracy: 1.0000
    Epoch 4366/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2557 - binary_accuracy: 1.0000
    Epoch 4367/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2556 - binary_accuracy: 1.0000
    Epoch 4368/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2556 - binary_accuracy: 1.0000
    Epoch 4369/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2556 - binary_accuracy: 1.0000
    Epoch 4370/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2555 - binary_accuracy: 1.0000
    Epoch 4371/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2555 - binary_accuracy: 1.0000
    Epoch 4372/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2555 - binary_accuracy: 1.0000
    Epoch 4373/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2554 - binary_accuracy: 1.0000
    Epoch 4374/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2554 - binary_accuracy: 1.0000
    Epoch 4375/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2554 - binary_accuracy: 1.0000
    Epoch 4376/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2553 - binary_accuracy: 1.0000
    Epoch 4377/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2553 - binary_accuracy: 1.0000
    Epoch 4378/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2553 - binary_accuracy: 1.0000
    Epoch 4379/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2552 - binary_accuracy: 1.0000
    Epoch 4380/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2552 - binary_accuracy: 1.0000
    Epoch 4381/7000
    1/1 [==============================] - 0s 50ms/step - loss: 0.2552 - binary_accuracy: 1.0000
    Epoch 4382/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2551 - binary_accuracy: 1.0000
    Epoch 4383/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2551 - binary_accuracy: 1.0000
    Epoch 4384/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2551 - binary_accuracy: 1.0000
    Epoch 4385/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2550 - binary_accuracy: 1.0000
    Epoch 4386/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2550 - binary_accuracy: 1.0000
    Epoch 4387/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2550 - binary_accuracy: 1.0000
    Epoch 4388/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2549 - binary_accuracy: 1.0000
    Epoch 4389/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2549 - binary_accuracy: 1.0000
    Epoch 4390/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2549 - binary_accuracy: 1.0000
    Epoch 4391/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2548 - binary_accuracy: 1.0000
    Epoch 4392/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2548 - binary_accuracy: 1.0000
    Epoch 4393/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2547 - binary_accuracy: 1.0000
    Epoch 4394/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2547 - binary_accuracy: 1.0000
    Epoch 4395/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2547 - binary_accuracy: 1.0000
    Epoch 4396/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2546 - binary_accuracy: 1.0000
    Epoch 4397/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2546 - binary_accuracy: 1.0000
    Epoch 4398/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2546 - binary_accuracy: 1.0000
    Epoch 4399/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2545 - binary_accuracy: 1.0000
    Epoch 4400/7000
    1/1 [==============================] - 0s 22ms/step - loss: 0.2545 - binary_accuracy: 1.0000
    Epoch 4401/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2545 - binary_accuracy: 1.0000
    Epoch 4402/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2544 - binary_accuracy: 1.0000
    Epoch 4403/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2544 - binary_accuracy: 1.0000
    Epoch 4404/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2544 - binary_accuracy: 1.0000
    Epoch 4405/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2543 - binary_accuracy: 1.0000
    Epoch 4406/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2543 - binary_accuracy: 1.0000
    Epoch 4407/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2543 - binary_accuracy: 1.0000
    Epoch 4408/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2542 - binary_accuracy: 1.0000
    Epoch 4409/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2542 - binary_accuracy: 1.0000
    Epoch 4410/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.2542 - binary_accuracy: 1.0000
    Epoch 4411/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2541 - binary_accuracy: 1.0000
    Epoch 4412/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2541 - binary_accuracy: 1.0000
    Epoch 4413/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2541 - binary_accuracy: 1.0000
    Epoch 4414/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2540 - binary_accuracy: 1.0000
    Epoch 4415/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2540 - binary_accuracy: 1.0000
    Epoch 4416/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2540 - binary_accuracy: 1.0000
    Epoch 4417/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2539 - binary_accuracy: 1.0000
    Epoch 4418/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2539 - binary_accuracy: 1.0000
    Epoch 4419/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2539 - binary_accuracy: 1.0000
    Epoch 4420/7000
    1/1 [==============================] - 0s 15ms/step - loss: 0.2538 - binary_accuracy: 1.0000
    Epoch 4421/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2538 - binary_accuracy: 1.0000
    Epoch 4422/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2538 - binary_accuracy: 1.0000
    Epoch 4423/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2537 - binary_accuracy: 1.0000
    Epoch 4424/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2537 - binary_accuracy: 1.0000
    Epoch 4425/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2537 - binary_accuracy: 1.0000
    Epoch 4426/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2536 - binary_accuracy: 1.0000
    Epoch 4427/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2536 - binary_accuracy: 1.0000
    Epoch 4428/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2535 - binary_accuracy: 1.0000
    Epoch 4429/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2535 - binary_accuracy: 1.0000
    Epoch 4430/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2535 - binary_accuracy: 1.0000
    Epoch 4431/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2534 - binary_accuracy: 1.0000
    Epoch 4432/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2534 - binary_accuracy: 1.0000
    Epoch 4433/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2534 - binary_accuracy: 1.0000
    Epoch 4434/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2533 - binary_accuracy: 1.0000
    Epoch 4435/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2533 - binary_accuracy: 1.0000
    Epoch 4436/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2533 - binary_accuracy: 1.0000
    Epoch 4437/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2532 - binary_accuracy: 1.0000
    Epoch 4438/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2532 - binary_accuracy: 1.0000
    Epoch 4439/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.2532 - binary_accuracy: 1.0000
    Epoch 4440/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2531 - binary_accuracy: 1.0000
    Epoch 4441/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2531 - binary_accuracy: 1.0000
    Epoch 4442/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2531 - binary_accuracy: 1.0000
    Epoch 4443/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2530 - binary_accuracy: 1.0000
    Epoch 4444/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2530 - binary_accuracy: 1.0000
    Epoch 4445/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2530 - binary_accuracy: 1.0000
    Epoch 4446/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2529 - binary_accuracy: 1.0000
    Epoch 4447/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2529 - binary_accuracy: 1.0000
    Epoch 4448/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2529 - binary_accuracy: 1.0000
    Epoch 4449/7000
    1/1 [==============================] - 0s 46ms/step - loss: 0.2528 - binary_accuracy: 1.0000
    Epoch 4450/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2528 - binary_accuracy: 1.0000
    Epoch 4451/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2528 - binary_accuracy: 1.0000
    Epoch 4452/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2527 - binary_accuracy: 1.0000
    Epoch 4453/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2527 - binary_accuracy: 1.0000
    Epoch 4454/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2527 - binary_accuracy: 1.0000
    Epoch 4455/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2526 - binary_accuracy: 1.0000
    Epoch 4456/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2526 - binary_accuracy: 1.0000
    Epoch 4457/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2526 - binary_accuracy: 1.0000
    Epoch 4458/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2525 - binary_accuracy: 1.0000
    Epoch 4459/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.2525 - binary_accuracy: 1.0000
    Epoch 4460/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2525 - binary_accuracy: 1.0000
    Epoch 4461/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2524 - binary_accuracy: 1.0000
    Epoch 4462/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2524 - binary_accuracy: 1.0000
    Epoch 4463/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2524 - binary_accuracy: 1.0000
    Epoch 4464/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2523 - binary_accuracy: 1.0000
    Epoch 4465/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2523 - binary_accuracy: 1.0000
    Epoch 4466/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2523 - binary_accuracy: 1.0000
    Epoch 4467/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2522 - binary_accuracy: 1.0000
    Epoch 4468/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2522 - binary_accuracy: 1.0000
    Epoch 4469/7000
    1/1 [==============================] - 0s 19ms/step - loss: 0.2522 - binary_accuracy: 1.0000
    Epoch 4470/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2521 - binary_accuracy: 1.0000
    Epoch 4471/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2521 - binary_accuracy: 1.0000
    Epoch 4472/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2521 - binary_accuracy: 1.0000
    Epoch 4473/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2520 - binary_accuracy: 1.0000
    Epoch 4474/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2520 - binary_accuracy: 1.0000
    Epoch 4475/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2520 - binary_accuracy: 1.0000
    Epoch 4476/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2519 - binary_accuracy: 1.0000
    Epoch 4477/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2519 - binary_accuracy: 1.0000
    Epoch 4478/7000
    1/1 [==============================] - 0s 26ms/step - loss: 0.2519 - binary_accuracy: 1.0000
    Epoch 4479/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2518 - binary_accuracy: 1.0000
    Epoch 4480/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2518 - binary_accuracy: 1.0000
    Epoch 4481/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2518 - binary_accuracy: 1.0000
    Epoch 4482/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2517 - binary_accuracy: 1.0000
    Epoch 4483/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2517 - binary_accuracy: 1.0000
    Epoch 4484/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2517 - binary_accuracy: 1.0000
    Epoch 4485/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2516 - binary_accuracy: 1.0000
    Epoch 4486/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2516 - binary_accuracy: 1.0000
    Epoch 4487/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2516 - binary_accuracy: 1.0000
    Epoch 4488/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2515 - binary_accuracy: 1.0000
    Epoch 4489/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2515 - binary_accuracy: 1.0000
    Epoch 4490/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2515 - binary_accuracy: 1.0000
    Epoch 4491/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2514 - binary_accuracy: 1.0000
    Epoch 4492/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2514 - binary_accuracy: 1.0000
    Epoch 4493/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2514 - binary_accuracy: 1.0000
    Epoch 4494/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2513 - binary_accuracy: 1.0000
    Epoch 4495/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2513 - binary_accuracy: 1.0000
    Epoch 4496/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2513 - binary_accuracy: 1.0000
    Epoch 4497/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2512 - binary_accuracy: 1.0000
    Epoch 4498/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2512 - binary_accuracy: 1.0000
    Epoch 4499/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2512 - binary_accuracy: 1.0000
    Epoch 4500/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2511 - binary_accuracy: 1.0000
    Epoch 4501/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2511 - binary_accuracy: 1.0000
    Epoch 4502/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2511 - binary_accuracy: 1.0000
    Epoch 4503/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2510 - binary_accuracy: 1.0000
    Epoch 4504/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2510 - binary_accuracy: 1.0000
    Epoch 4505/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2510 - binary_accuracy: 1.0000
    Epoch 4506/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2509 - binary_accuracy: 1.0000
    Epoch 4507/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2509 - binary_accuracy: 1.0000
    Epoch 4508/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.2509 - binary_accuracy: 1.0000
    Epoch 4509/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2508 - binary_accuracy: 1.0000
    Epoch 4510/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2508 - binary_accuracy: 1.0000
    Epoch 4511/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2508 - binary_accuracy: 1.0000
    Epoch 4512/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2507 - binary_accuracy: 1.0000
    Epoch 4513/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2507 - binary_accuracy: 1.0000
    Epoch 4514/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2507 - binary_accuracy: 1.0000
    Epoch 4515/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2506 - binary_accuracy: 1.0000
    Epoch 4516/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2506 - binary_accuracy: 1.0000
    Epoch 4517/7000
    1/1 [==============================] - 0s 32ms/step - loss: 0.2506 - binary_accuracy: 1.0000
    Epoch 4518/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2505 - binary_accuracy: 1.0000
    Epoch 4519/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2505 - binary_accuracy: 1.0000
    Epoch 4520/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2505 - binary_accuracy: 1.0000
    Epoch 4521/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2504 - binary_accuracy: 1.0000
    Epoch 4522/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2504 - binary_accuracy: 1.0000
    Epoch 4523/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2504 - binary_accuracy: 1.0000
    Epoch 4524/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2503 - binary_accuracy: 1.0000
    Epoch 4525/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2503 - binary_accuracy: 1.0000
    Epoch 4526/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2503 - binary_accuracy: 1.0000
    Epoch 4527/7000
    1/1 [==============================] - 0s 42ms/step - loss: 0.2502 - binary_accuracy: 1.0000
    Epoch 4528/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2502 - binary_accuracy: 1.0000
    Epoch 4529/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2502 - binary_accuracy: 1.0000
    Epoch 4530/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2501 - binary_accuracy: 1.0000
    Epoch 4531/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2501 - binary_accuracy: 1.0000
    Epoch 4532/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2501 - binary_accuracy: 1.0000
    Epoch 4533/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2500 - binary_accuracy: 1.0000
    Epoch 4534/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2500 - binary_accuracy: 1.0000
    Epoch 4535/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2500 - binary_accuracy: 1.0000
    Epoch 4536/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2499 - binary_accuracy: 1.0000
    Epoch 4537/7000
    1/1 [==============================] - 0s 35ms/step - loss: 0.2499 - binary_accuracy: 1.0000
    Epoch 4538/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2499 - binary_accuracy: 1.0000
    Epoch 4539/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2498 - binary_accuracy: 1.0000
    Epoch 4540/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2498 - binary_accuracy: 1.0000
    Epoch 4541/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2498 - binary_accuracy: 1.0000
    Epoch 4542/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2497 - binary_accuracy: 1.0000
    Epoch 4543/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2497 - binary_accuracy: 1.0000
    Epoch 4544/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2497 - binary_accuracy: 1.0000
    Epoch 4545/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2496 - binary_accuracy: 1.0000
    Epoch 4546/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2496 - binary_accuracy: 1.0000
    Epoch 4547/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2496 - binary_accuracy: 1.0000
    Epoch 4548/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2495 - binary_accuracy: 1.0000
    Epoch 4549/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2495 - binary_accuracy: 1.0000
    Epoch 4550/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2495 - binary_accuracy: 1.0000
    Epoch 4551/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2494 - binary_accuracy: 1.0000
    Epoch 4552/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2494 - binary_accuracy: 1.0000
    Epoch 4553/7000
    1/1 [==============================] - 0s 26ms/step - loss: 0.2494 - binary_accuracy: 1.0000
    Epoch 4554/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2493 - binary_accuracy: 1.0000
    Epoch 4555/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2493 - binary_accuracy: 1.0000
    Epoch 4556/7000
    1/1 [==============================] - 0s 56ms/step - loss: 0.2493 - binary_accuracy: 1.0000
    Epoch 4557/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2492 - binary_accuracy: 1.0000
    Epoch 4558/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2492 - binary_accuracy: 1.0000
    Epoch 4559/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2492 - binary_accuracy: 1.0000
    Epoch 4560/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2491 - binary_accuracy: 1.0000
    Epoch 4561/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2491 - binary_accuracy: 1.0000
    Epoch 4562/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2491 - binary_accuracy: 1.0000
    Epoch 4563/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2490 - binary_accuracy: 1.0000
    Epoch 4564/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2490 - binary_accuracy: 1.0000
    Epoch 4565/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2490 - binary_accuracy: 1.0000
    Epoch 4566/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2489 - binary_accuracy: 1.0000
    Epoch 4567/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2489 - binary_accuracy: 1.0000
    Epoch 4568/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2489 - binary_accuracy: 1.0000
    Epoch 4569/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2488 - binary_accuracy: 1.0000
    Epoch 4570/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2488 - binary_accuracy: 1.0000
    Epoch 4571/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2488 - binary_accuracy: 1.0000
    Epoch 4572/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2487 - binary_accuracy: 1.0000
    Epoch 4573/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2487 - binary_accuracy: 1.0000
    Epoch 4574/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2487 - binary_accuracy: 1.0000
    Epoch 4575/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2486 - binary_accuracy: 1.0000
    Epoch 4576/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2486 - binary_accuracy: 1.0000
    Epoch 4577/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2486 - binary_accuracy: 1.0000
    Epoch 4578/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2485 - binary_accuracy: 1.0000
    Epoch 4579/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2485 - binary_accuracy: 1.0000
    Epoch 4580/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2485 - binary_accuracy: 1.0000
    Epoch 4581/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2484 - binary_accuracy: 1.0000
    Epoch 4582/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2484 - binary_accuracy: 1.0000
    Epoch 4583/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2484 - binary_accuracy: 1.0000
    Epoch 4584/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2483 - binary_accuracy: 1.0000
    Epoch 4585/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2483 - binary_accuracy: 1.0000
    Epoch 4586/7000
    1/1 [==============================] - 0s 25ms/step - loss: 0.2483 - binary_accuracy: 1.0000
    Epoch 4587/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2482 - binary_accuracy: 1.0000
    Epoch 4588/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2482 - binary_accuracy: 1.0000
    Epoch 4589/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2482 - binary_accuracy: 1.0000
    Epoch 4590/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2481 - binary_accuracy: 1.0000
    Epoch 4591/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2481 - binary_accuracy: 1.0000
    Epoch 4592/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2481 - binary_accuracy: 1.0000
    Epoch 4593/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2481 - binary_accuracy: 1.0000
    Epoch 4594/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2480 - binary_accuracy: 1.0000
    Epoch 4595/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2480 - binary_accuracy: 1.0000
    Epoch 4596/7000
    1/1 [==============================] - 0s 31ms/step - loss: 0.2480 - binary_accuracy: 1.0000
    Epoch 4597/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2479 - binary_accuracy: 1.0000
    Epoch 4598/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2479 - binary_accuracy: 1.0000
    Epoch 4599/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2479 - binary_accuracy: 1.0000
    Epoch 4600/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2478 - binary_accuracy: 1.0000
    Epoch 4601/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2478 - binary_accuracy: 1.0000
    Epoch 4602/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2478 - binary_accuracy: 1.0000
    Epoch 4603/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2477 - binary_accuracy: 1.0000
    Epoch 4604/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2477 - binary_accuracy: 1.0000
    Epoch 4605/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.2477 - binary_accuracy: 1.0000
    Epoch 4606/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2476 - binary_accuracy: 1.0000
    Epoch 4607/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2476 - binary_accuracy: 1.0000
    Epoch 4608/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2476 - binary_accuracy: 1.0000
    Epoch 4609/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2475 - binary_accuracy: 1.0000
    Epoch 4610/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2475 - binary_accuracy: 1.0000
    Epoch 4611/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2475 - binary_accuracy: 1.0000
    Epoch 4612/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2474 - binary_accuracy: 1.0000
    Epoch 4613/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2474 - binary_accuracy: 1.0000
    Epoch 4614/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2474 - binary_accuracy: 1.0000
    Epoch 4615/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2473 - binary_accuracy: 1.0000
    Epoch 4616/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2473 - binary_accuracy: 1.0000
    Epoch 4617/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2473 - binary_accuracy: 1.0000
    Epoch 4618/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2472 - binary_accuracy: 1.0000
    Epoch 4619/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2472 - binary_accuracy: 1.0000
    Epoch 4620/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2472 - binary_accuracy: 1.0000
    Epoch 4621/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2471 - binary_accuracy: 1.0000
    Epoch 4622/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2471 - binary_accuracy: 1.0000
    Epoch 4623/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2471 - binary_accuracy: 1.0000
    Epoch 4624/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2470 - binary_accuracy: 1.0000
    Epoch 4625/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2470 - binary_accuracy: 1.0000
    Epoch 4626/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2470 - binary_accuracy: 1.0000
    Epoch 4627/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2470 - binary_accuracy: 1.0000
    Epoch 4628/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2469 - binary_accuracy: 1.0000
    Epoch 4629/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2469 - binary_accuracy: 1.0000
    Epoch 4630/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2469 - binary_accuracy: 1.0000
    Epoch 4631/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2468 - binary_accuracy: 1.0000
    Epoch 4632/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2468 - binary_accuracy: 1.0000
    Epoch 4633/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2468 - binary_accuracy: 1.0000
    Epoch 4634/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2467 - binary_accuracy: 1.0000
    Epoch 4635/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.2467 - binary_accuracy: 1.0000
    Epoch 4636/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2467 - binary_accuracy: 1.0000
    Epoch 4637/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2466 - binary_accuracy: 1.0000
    Epoch 4638/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2466 - binary_accuracy: 1.0000
    Epoch 4639/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2466 - binary_accuracy: 1.0000
    Epoch 4640/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2465 - binary_accuracy: 1.0000
    Epoch 4641/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2465 - binary_accuracy: 1.0000
    Epoch 4642/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2465 - binary_accuracy: 1.0000
    Epoch 4643/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2464 - binary_accuracy: 1.0000
    Epoch 4644/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2464 - binary_accuracy: 1.0000
    Epoch 4645/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2464 - binary_accuracy: 1.0000
    Epoch 4646/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2463 - binary_accuracy: 1.0000
    Epoch 4647/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2463 - binary_accuracy: 1.0000
    Epoch 4648/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2463 - binary_accuracy: 1.0000
    Epoch 4649/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2462 - binary_accuracy: 1.0000
    Epoch 4650/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2462 - binary_accuracy: 1.0000
    Epoch 4651/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2462 - binary_accuracy: 1.0000
    Epoch 4652/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2461 - binary_accuracy: 1.0000
    Epoch 4653/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2461 - binary_accuracy: 1.0000
    Epoch 4654/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2461 - binary_accuracy: 1.0000
    Epoch 4655/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2461 - binary_accuracy: 1.0000
    Epoch 4656/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2460 - binary_accuracy: 1.0000
    Epoch 4657/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2460 - binary_accuracy: 1.0000
    Epoch 4658/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2460 - binary_accuracy: 1.0000
    Epoch 4659/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2459 - binary_accuracy: 1.0000
    Epoch 4660/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2459 - binary_accuracy: 1.0000
    Epoch 4661/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2459 - binary_accuracy: 1.0000
    Epoch 4662/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2458 - binary_accuracy: 1.0000
    Epoch 4663/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2458 - binary_accuracy: 1.0000
    Epoch 4664/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2458 - binary_accuracy: 1.0000
    Epoch 4665/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2457 - binary_accuracy: 1.0000
    Epoch 4666/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2457 - binary_accuracy: 1.0000
    Epoch 4667/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2457 - binary_accuracy: 1.0000
    Epoch 4668/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2456 - binary_accuracy: 1.0000
    Epoch 4669/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2456 - binary_accuracy: 1.0000
    Epoch 4670/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2456 - binary_accuracy: 1.0000
    Epoch 4671/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2455 - binary_accuracy: 1.0000
    Epoch 4672/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2455 - binary_accuracy: 1.0000
    Epoch 4673/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2455 - binary_accuracy: 1.0000
    Epoch 4674/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2454 - binary_accuracy: 1.0000
    Epoch 4675/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2454 - binary_accuracy: 1.0000
    Epoch 4676/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2454 - binary_accuracy: 1.0000
    Epoch 4677/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2454 - binary_accuracy: 1.0000
    Epoch 4678/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2453 - binary_accuracy: 1.0000
    Epoch 4679/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2453 - binary_accuracy: 1.0000
    Epoch 4680/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2453 - binary_accuracy: 1.0000
    Epoch 4681/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2452 - binary_accuracy: 1.0000
    Epoch 4682/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2452 - binary_accuracy: 1.0000
    Epoch 4683/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2452 - binary_accuracy: 1.0000
    Epoch 4684/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2451 - binary_accuracy: 1.0000
    Epoch 4685/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2451 - binary_accuracy: 1.0000
    Epoch 4686/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2451 - binary_accuracy: 1.0000
    Epoch 4687/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2450 - binary_accuracy: 1.0000
    Epoch 4688/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2450 - binary_accuracy: 1.0000
    Epoch 4689/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2450 - binary_accuracy: 1.0000
    Epoch 4690/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2449 - binary_accuracy: 1.0000
    Epoch 4691/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2449 - binary_accuracy: 1.0000
    Epoch 4692/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2449 - binary_accuracy: 1.0000
    Epoch 4693/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2448 - binary_accuracy: 1.0000
    Epoch 4694/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.2448 - binary_accuracy: 1.0000
    Epoch 4695/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2448 - binary_accuracy: 1.0000
    Epoch 4696/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2447 - binary_accuracy: 1.0000
    Epoch 4697/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2447 - binary_accuracy: 1.0000
    Epoch 4698/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2447 - binary_accuracy: 1.0000
    Epoch 4699/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2447 - binary_accuracy: 1.0000
    Epoch 4700/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2446 - binary_accuracy: 1.0000
    Epoch 4701/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2446 - binary_accuracy: 1.0000
    Epoch 4702/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2446 - binary_accuracy: 1.0000
    Epoch 4703/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2445 - binary_accuracy: 1.0000
    Epoch 4704/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2445 - binary_accuracy: 1.0000
    Epoch 4705/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2445 - binary_accuracy: 1.0000
    Epoch 4706/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2444 - binary_accuracy: 1.0000
    Epoch 4707/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2444 - binary_accuracy: 1.0000
    Epoch 4708/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2444 - binary_accuracy: 1.0000
    Epoch 4709/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2443 - binary_accuracy: 1.0000
    Epoch 4710/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2443 - binary_accuracy: 1.0000
    Epoch 4711/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2443 - binary_accuracy: 1.0000
    Epoch 4712/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2442 - binary_accuracy: 1.0000
    Epoch 4713/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2442 - binary_accuracy: 1.0000
    Epoch 4714/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2442 - binary_accuracy: 1.0000
    Epoch 4715/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2441 - binary_accuracy: 1.0000
    Epoch 4716/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2441 - binary_accuracy: 1.0000
    Epoch 4717/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2441 - binary_accuracy: 1.0000
    Epoch 4718/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2441 - binary_accuracy: 1.0000
    Epoch 4719/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2440 - binary_accuracy: 1.0000
    Epoch 4720/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2440 - binary_accuracy: 1.0000
    Epoch 4721/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2440 - binary_accuracy: 1.0000
    Epoch 4722/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2439 - binary_accuracy: 1.0000
    Epoch 4723/7000
    1/1 [==============================] - 0s 22ms/step - loss: 0.2439 - binary_accuracy: 1.0000
    Epoch 4724/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2439 - binary_accuracy: 1.0000
    Epoch 4725/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2438 - binary_accuracy: 1.0000
    Epoch 4726/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2438 - binary_accuracy: 1.0000
    Epoch 4727/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2438 - binary_accuracy: 1.0000
    Epoch 4728/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2437 - binary_accuracy: 1.0000
    Epoch 4729/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2437 - binary_accuracy: 1.0000
    Epoch 4730/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2437 - binary_accuracy: 1.0000
    Epoch 4731/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2436 - binary_accuracy: 1.0000
    Epoch 4732/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2436 - binary_accuracy: 1.0000
    Epoch 4733/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2436 - binary_accuracy: 1.0000
    Epoch 4734/7000
    1/1 [==============================] - 0s 17ms/step - loss: 0.2436 - binary_accuracy: 1.0000
    Epoch 4735/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2435 - binary_accuracy: 1.0000
    Epoch 4736/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2435 - binary_accuracy: 1.0000
    Epoch 4737/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2435 - binary_accuracy: 1.0000
    Epoch 4738/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2434 - binary_accuracy: 1.0000
    Epoch 4739/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2434 - binary_accuracy: 1.0000
    Epoch 4740/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2434 - binary_accuracy: 1.0000
    Epoch 4741/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2433 - binary_accuracy: 1.0000
    Epoch 4742/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2433 - binary_accuracy: 1.0000
    Epoch 4743/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2433 - binary_accuracy: 1.0000
    Epoch 4744/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2432 - binary_accuracy: 1.0000
    Epoch 4745/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2432 - binary_accuracy: 1.0000
    Epoch 4746/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2432 - binary_accuracy: 1.0000
    Epoch 4747/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2431 - binary_accuracy: 1.0000
    Epoch 4748/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2431 - binary_accuracy: 1.0000
    Epoch 4749/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2431 - binary_accuracy: 1.0000
    Epoch 4750/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2431 - binary_accuracy: 1.0000
    Epoch 4751/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2430 - binary_accuracy: 1.0000
    Epoch 4752/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2430 - binary_accuracy: 1.0000
    Epoch 4753/7000
    1/1 [==============================] - 0s 21ms/step - loss: 0.2430 - binary_accuracy: 1.0000
    Epoch 4754/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2429 - binary_accuracy: 1.0000
    Epoch 4755/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2429 - binary_accuracy: 1.0000
    Epoch 4756/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2429 - binary_accuracy: 1.0000
    Epoch 4757/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2428 - binary_accuracy: 1.0000
    Epoch 4758/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2428 - binary_accuracy: 1.0000
    Epoch 4759/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2428 - binary_accuracy: 1.0000
    Epoch 4760/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2427 - binary_accuracy: 1.0000
    Epoch 4761/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2427 - binary_accuracy: 1.0000
    Epoch 4762/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2427 - binary_accuracy: 1.0000
    Epoch 4763/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.2426 - binary_accuracy: 1.0000
    Epoch 4764/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2426 - binary_accuracy: 1.0000
    Epoch 4765/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2426 - binary_accuracy: 1.0000
    Epoch 4766/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2426 - binary_accuracy: 1.0000
    Epoch 4767/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2425 - binary_accuracy: 1.0000
    Epoch 4768/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2425 - binary_accuracy: 1.0000
    Epoch 4769/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2425 - binary_accuracy: 1.0000
    Epoch 4770/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2424 - binary_accuracy: 1.0000
    Epoch 4771/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2424 - binary_accuracy: 1.0000
    Epoch 4772/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2424 - binary_accuracy: 1.0000
    Epoch 4773/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.2423 - binary_accuracy: 1.0000
    Epoch 4774/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2423 - binary_accuracy: 1.0000
    Epoch 4775/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2423 - binary_accuracy: 1.0000
    Epoch 4776/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2422 - binary_accuracy: 1.0000
    Epoch 4777/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2422 - binary_accuracy: 1.0000
    Epoch 4778/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2422 - binary_accuracy: 1.0000
    Epoch 4779/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2422 - binary_accuracy: 1.0000
    Epoch 4780/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2421 - binary_accuracy: 1.0000
    Epoch 4781/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2421 - binary_accuracy: 1.0000
    Epoch 4782/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2421 - binary_accuracy: 1.0000
    Epoch 4783/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2420 - binary_accuracy: 1.0000
    Epoch 4784/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2420 - binary_accuracy: 1.0000
    Epoch 4785/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2420 - binary_accuracy: 1.0000
    Epoch 4786/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2419 - binary_accuracy: 1.0000
    Epoch 4787/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2419 - binary_accuracy: 1.0000
    Epoch 4788/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2419 - binary_accuracy: 1.0000
    Epoch 4789/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2418 - binary_accuracy: 1.0000
    Epoch 4790/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2418 - binary_accuracy: 1.0000
    Epoch 4791/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2418 - binary_accuracy: 1.0000
    Epoch 4792/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2418 - binary_accuracy: 1.0000
    Epoch 4793/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2417 - binary_accuracy: 1.0000
    Epoch 4794/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2417 - binary_accuracy: 1.0000
    Epoch 4795/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2417 - binary_accuracy: 1.0000
    Epoch 4796/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2416 - binary_accuracy: 1.0000
    Epoch 4797/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2416 - binary_accuracy: 1.0000
    Epoch 4798/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2416 - binary_accuracy: 1.0000
    Epoch 4799/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2415 - binary_accuracy: 1.0000
    Epoch 4800/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2415 - binary_accuracy: 1.0000
    Epoch 4801/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2415 - binary_accuracy: 1.0000
    Epoch 4802/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2414 - binary_accuracy: 1.0000
    Epoch 4803/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2414 - binary_accuracy: 1.0000
    Epoch 4804/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2414 - binary_accuracy: 1.0000
    Epoch 4805/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2414 - binary_accuracy: 1.0000
    Epoch 4806/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2413 - binary_accuracy: 1.0000
    Epoch 4807/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2413 - binary_accuracy: 1.0000
    Epoch 4808/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2413 - binary_accuracy: 1.0000
    Epoch 4809/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2412 - binary_accuracy: 1.0000
    Epoch 4810/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2412 - binary_accuracy: 1.0000
    Epoch 4811/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.2412 - binary_accuracy: 1.0000
    Epoch 4812/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2411 - binary_accuracy: 1.0000
    Epoch 4813/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2411 - binary_accuracy: 1.0000
    Epoch 4814/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2411 - binary_accuracy: 1.0000
    Epoch 4815/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2410 - binary_accuracy: 1.0000
    Epoch 4816/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2410 - binary_accuracy: 1.0000
    Epoch 4817/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2410 - binary_accuracy: 1.0000
    Epoch 4818/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2410 - binary_accuracy: 1.0000
    Epoch 4819/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2409 - binary_accuracy: 1.0000
    Epoch 4820/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2409 - binary_accuracy: 1.0000
    Epoch 4821/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2409 - binary_accuracy: 1.0000
    Epoch 4822/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2408 - binary_accuracy: 1.0000
    Epoch 4823/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2408 - binary_accuracy: 1.0000
    Epoch 4824/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2408 - binary_accuracy: 1.0000
    Epoch 4825/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2407 - binary_accuracy: 1.0000
    Epoch 4826/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2407 - binary_accuracy: 1.0000
    Epoch 4827/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2407 - binary_accuracy: 1.0000
    Epoch 4828/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2406 - binary_accuracy: 1.0000
    Epoch 4829/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2406 - binary_accuracy: 1.0000
    Epoch 4830/7000
    1/1 [==============================] - 0s 23ms/step - loss: 0.2406 - binary_accuracy: 1.0000
    Epoch 4831/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2406 - binary_accuracy: 1.0000
    Epoch 4832/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2405 - binary_accuracy: 1.0000
    Epoch 4833/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2405 - binary_accuracy: 1.0000
    Epoch 4834/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2405 - binary_accuracy: 1.0000
    Epoch 4835/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2404 - binary_accuracy: 1.0000
    Epoch 4836/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2404 - binary_accuracy: 1.0000
    Epoch 4837/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2404 - binary_accuracy: 1.0000
    Epoch 4838/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2403 - binary_accuracy: 1.0000
    Epoch 4839/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2403 - binary_accuracy: 1.0000
    Epoch 4840/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2403 - binary_accuracy: 1.0000
    Epoch 4841/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2403 - binary_accuracy: 1.0000
    Epoch 4842/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2402 - binary_accuracy: 1.0000
    Epoch 4843/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2402 - binary_accuracy: 1.0000
    Epoch 4844/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2402 - binary_accuracy: 1.0000
    Epoch 4845/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2401 - binary_accuracy: 1.0000
    Epoch 4846/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2401 - binary_accuracy: 1.0000
    Epoch 4847/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2401 - binary_accuracy: 1.0000
    Epoch 4848/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2400 - binary_accuracy: 1.0000
    Epoch 4849/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2400 - binary_accuracy: 1.0000
    Epoch 4850/7000
    1/1 [==============================] - 0s 32ms/step - loss: 0.2400 - binary_accuracy: 1.0000
    Epoch 4851/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2399 - binary_accuracy: 1.0000
    Epoch 4852/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2399 - binary_accuracy: 1.0000
    Epoch 4853/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2399 - binary_accuracy: 1.0000
    Epoch 4854/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2399 - binary_accuracy: 1.0000
    Epoch 4855/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2398 - binary_accuracy: 1.0000
    Epoch 4856/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2398 - binary_accuracy: 1.0000
    Epoch 4857/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2398 - binary_accuracy: 1.0000
    Epoch 4858/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2397 - binary_accuracy: 1.0000
    Epoch 4859/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2397 - binary_accuracy: 1.0000
    Epoch 4860/7000
    1/1 [==============================] - 0s 24ms/step - loss: 0.2397 - binary_accuracy: 1.0000
    Epoch 4861/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2396 - binary_accuracy: 1.0000
    Epoch 4862/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2396 - binary_accuracy: 1.0000
    Epoch 4863/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2396 - binary_accuracy: 1.0000
    Epoch 4864/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2396 - binary_accuracy: 1.0000
    Epoch 4865/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2395 - binary_accuracy: 1.0000
    Epoch 4866/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2395 - binary_accuracy: 1.0000
    Epoch 4867/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2395 - binary_accuracy: 1.0000
    Epoch 4868/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2394 - binary_accuracy: 1.0000
    Epoch 4869/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2394 - binary_accuracy: 1.0000
    Epoch 4870/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2394 - binary_accuracy: 1.0000
    Epoch 4871/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2393 - binary_accuracy: 1.0000
    Epoch 4872/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2393 - binary_accuracy: 1.0000
    Epoch 4873/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2393 - binary_accuracy: 1.0000
    Epoch 4874/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2393 - binary_accuracy: 1.0000
    Epoch 4875/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2392 - binary_accuracy: 1.0000
    Epoch 4876/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2392 - binary_accuracy: 1.0000
    Epoch 4877/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2392 - binary_accuracy: 1.0000
    Epoch 4878/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2391 - binary_accuracy: 1.0000
    Epoch 4879/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2391 - binary_accuracy: 1.0000
    Epoch 4880/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2391 - binary_accuracy: 1.0000
    Epoch 4881/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2390 - binary_accuracy: 1.0000
    Epoch 4882/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2390 - binary_accuracy: 1.0000
    Epoch 4883/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2390 - binary_accuracy: 1.0000
    Epoch 4884/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2390 - binary_accuracy: 1.0000
    Epoch 4885/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2389 - binary_accuracy: 1.0000
    Epoch 4886/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2389 - binary_accuracy: 1.0000
    Epoch 4887/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2389 - binary_accuracy: 1.0000
    Epoch 4888/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2388 - binary_accuracy: 1.0000
    Epoch 4889/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2388 - binary_accuracy: 1.0000
    Epoch 4890/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2388 - binary_accuracy: 1.0000
    Epoch 4891/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2387 - binary_accuracy: 1.0000
    Epoch 4892/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2387 - binary_accuracy: 1.0000
    Epoch 4893/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2387 - binary_accuracy: 1.0000
    Epoch 4894/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2387 - binary_accuracy: 1.0000
    Epoch 4895/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2386 - binary_accuracy: 1.0000
    Epoch 4896/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2386 - binary_accuracy: 1.0000
    Epoch 4897/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2386 - binary_accuracy: 1.0000
    Epoch 4898/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2385 - binary_accuracy: 1.0000
    Epoch 4899/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2385 - binary_accuracy: 1.0000
    Epoch 4900/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2385 - binary_accuracy: 1.0000
    Epoch 4901/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2384 - binary_accuracy: 1.0000
    Epoch 4902/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2384 - binary_accuracy: 1.0000
    Epoch 4903/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2384 - binary_accuracy: 1.0000
    Epoch 4904/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2384 - binary_accuracy: 1.0000
    Epoch 4905/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2383 - binary_accuracy: 1.0000
    Epoch 4906/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2383 - binary_accuracy: 1.0000
    Epoch 4907/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2383 - binary_accuracy: 1.0000
    Epoch 4908/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2382 - binary_accuracy: 1.0000
    Epoch 4909/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2382 - binary_accuracy: 1.0000
    Epoch 4910/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2382 - binary_accuracy: 1.0000
    Epoch 4911/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2381 - binary_accuracy: 1.0000
    Epoch 4912/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2381 - binary_accuracy: 1.0000
    Epoch 4913/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2381 - binary_accuracy: 1.0000
    Epoch 4914/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2381 - binary_accuracy: 1.0000
    Epoch 4915/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2380 - binary_accuracy: 1.0000
    Epoch 4916/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2380 - binary_accuracy: 1.0000
    Epoch 4917/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2380 - binary_accuracy: 1.0000
    Epoch 4918/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2379 - binary_accuracy: 1.0000
    Epoch 4919/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2379 - binary_accuracy: 1.0000
    Epoch 4920/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2379 - binary_accuracy: 1.0000
    Epoch 4921/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2378 - binary_accuracy: 1.0000
    Epoch 4922/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2378 - binary_accuracy: 1.0000
    Epoch 4923/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2378 - binary_accuracy: 1.0000
    Epoch 4924/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2378 - binary_accuracy: 1.0000
    Epoch 4925/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2377 - binary_accuracy: 1.0000
    Epoch 4926/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2377 - binary_accuracy: 1.0000
    Epoch 4927/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2377 - binary_accuracy: 1.0000
    Epoch 4928/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2376 - binary_accuracy: 1.0000
    Epoch 4929/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2376 - binary_accuracy: 1.0000
    Epoch 4930/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2376 - binary_accuracy: 1.0000
    Epoch 4931/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2376 - binary_accuracy: 1.0000
    Epoch 4932/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2375 - binary_accuracy: 1.0000
    Epoch 4933/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2375 - binary_accuracy: 1.0000
    Epoch 4934/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2375 - binary_accuracy: 1.0000
    Epoch 4935/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2374 - binary_accuracy: 1.0000
    Epoch 4936/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2374 - binary_accuracy: 1.0000
    Epoch 4937/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2374 - binary_accuracy: 1.0000
    Epoch 4938/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2373 - binary_accuracy: 1.0000
    Epoch 4939/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2373 - binary_accuracy: 1.0000
    Epoch 4940/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2373 - binary_accuracy: 1.0000
    Epoch 4941/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2373 - binary_accuracy: 1.0000
    Epoch 4942/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2372 - binary_accuracy: 1.0000
    Epoch 4943/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2372 - binary_accuracy: 1.0000
    Epoch 4944/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2372 - binary_accuracy: 1.0000
    Epoch 4945/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2371 - binary_accuracy: 1.0000
    Epoch 4946/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2371 - binary_accuracy: 1.0000
    Epoch 4947/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2371 - binary_accuracy: 1.0000
    Epoch 4948/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2370 - binary_accuracy: 1.0000
    Epoch 4949/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2370 - binary_accuracy: 1.0000
    Epoch 4950/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2370 - binary_accuracy: 1.0000
    Epoch 4951/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2370 - binary_accuracy: 1.0000
    Epoch 4952/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2369 - binary_accuracy: 1.0000
    Epoch 4953/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2369 - binary_accuracy: 1.0000
    Epoch 4954/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2369 - binary_accuracy: 1.0000
    Epoch 4955/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2368 - binary_accuracy: 1.0000
    Epoch 4956/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2368 - binary_accuracy: 1.0000
    Epoch 4957/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2368 - binary_accuracy: 1.0000
    Epoch 4958/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2368 - binary_accuracy: 1.0000
    Epoch 4959/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2367 - binary_accuracy: 1.0000
    Epoch 4960/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2367 - binary_accuracy: 1.0000
    Epoch 4961/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2367 - binary_accuracy: 1.0000
    Epoch 4962/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2366 - binary_accuracy: 1.0000
    Epoch 4963/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2366 - binary_accuracy: 1.0000
    Epoch 4964/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2366 - binary_accuracy: 1.0000
    Epoch 4965/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2365 - binary_accuracy: 1.0000
    Epoch 4966/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2365 - binary_accuracy: 1.0000
    Epoch 4967/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2365 - binary_accuracy: 1.0000
    Epoch 4968/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2365 - binary_accuracy: 1.0000
    Epoch 4969/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2364 - binary_accuracy: 1.0000
    Epoch 4970/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2364 - binary_accuracy: 1.0000
    Epoch 4971/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2364 - binary_accuracy: 1.0000
    Epoch 4972/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2363 - binary_accuracy: 1.0000
    Epoch 4973/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2363 - binary_accuracy: 1.0000
    Epoch 4974/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2363 - binary_accuracy: 1.0000
    Epoch 4975/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2363 - binary_accuracy: 1.0000
    Epoch 4976/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2362 - binary_accuracy: 1.0000
    Epoch 4977/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2362 - binary_accuracy: 1.0000
    Epoch 4978/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2362 - binary_accuracy: 1.0000
    Epoch 4979/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2361 - binary_accuracy: 1.0000
    Epoch 4980/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2361 - binary_accuracy: 1.0000
    Epoch 4981/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2361 - binary_accuracy: 1.0000
    Epoch 4982/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2360 - binary_accuracy: 1.0000
    Epoch 4983/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2360 - binary_accuracy: 1.0000
    Epoch 4984/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2360 - binary_accuracy: 1.0000
    Epoch 4985/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2360 - binary_accuracy: 1.0000
    Epoch 4986/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2359 - binary_accuracy: 1.0000
    Epoch 4987/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2359 - binary_accuracy: 1.0000
    Epoch 4988/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2359 - binary_accuracy: 1.0000
    Epoch 4989/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2358 - binary_accuracy: 1.0000
    Epoch 4990/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2358 - binary_accuracy: 1.0000
    Epoch 4991/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2358 - binary_accuracy: 1.0000
    Epoch 4992/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2358 - binary_accuracy: 1.0000
    Epoch 4993/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2357 - binary_accuracy: 1.0000
    Epoch 4994/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2357 - binary_accuracy: 1.0000
    Epoch 4995/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2357 - binary_accuracy: 1.0000
    Epoch 4996/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2356 - binary_accuracy: 1.0000
    Epoch 4997/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2356 - binary_accuracy: 1.0000
    Epoch 4998/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2356 - binary_accuracy: 1.0000
    Epoch 4999/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2356 - binary_accuracy: 1.0000
    Epoch 5000/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2355 - binary_accuracy: 1.0000
    Epoch 5001/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2355 - binary_accuracy: 1.0000
    Epoch 5002/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2355 - binary_accuracy: 1.0000
    Epoch 5003/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2354 - binary_accuracy: 1.0000
    Epoch 5004/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2354 - binary_accuracy: 1.0000
    Epoch 5005/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2354 - binary_accuracy: 1.0000
    Epoch 5006/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2353 - binary_accuracy: 1.0000
    Epoch 5007/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2353 - binary_accuracy: 1.0000
    Epoch 5008/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2353 - binary_accuracy: 1.0000
    Epoch 5009/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2353 - binary_accuracy: 1.0000
    Epoch 5010/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2352 - binary_accuracy: 1.0000
    Epoch 5011/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2352 - binary_accuracy: 1.0000
    Epoch 5012/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2352 - binary_accuracy: 1.0000
    Epoch 5013/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2351 - binary_accuracy: 1.0000
    Epoch 5014/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2351 - binary_accuracy: 1.0000
    Epoch 5015/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2351 - binary_accuracy: 1.0000
    Epoch 5016/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2351 - binary_accuracy: 1.0000
    Epoch 5017/7000
    1/1 [==============================] - 0s 17ms/step - loss: 0.2350 - binary_accuracy: 1.0000
    Epoch 5018/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2350 - binary_accuracy: 1.0000
    Epoch 5019/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2350 - binary_accuracy: 1.0000
    Epoch 5020/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2349 - binary_accuracy: 1.0000
    Epoch 5021/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2349 - binary_accuracy: 1.0000
    Epoch 5022/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2349 - binary_accuracy: 1.0000
    Epoch 5023/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2349 - binary_accuracy: 1.0000
    Epoch 5024/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2348 - binary_accuracy: 1.0000
    Epoch 5025/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2348 - binary_accuracy: 1.0000
    Epoch 5026/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2348 - binary_accuracy: 1.0000
    Epoch 5027/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2347 - binary_accuracy: 1.0000
    Epoch 5028/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2347 - binary_accuracy: 1.0000
    Epoch 5029/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2347 - binary_accuracy: 1.0000
    Epoch 5030/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2347 - binary_accuracy: 1.0000
    Epoch 5031/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2346 - binary_accuracy: 1.0000
    Epoch 5032/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2346 - binary_accuracy: 1.0000
    Epoch 5033/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2346 - binary_accuracy: 1.0000
    Epoch 5034/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2345 - binary_accuracy: 1.0000
    Epoch 5035/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2345 - binary_accuracy: 1.0000
    Epoch 5036/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2345 - binary_accuracy: 1.0000
    Epoch 5037/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2344 - binary_accuracy: 1.0000
    Epoch 5038/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2344 - binary_accuracy: 1.0000
    Epoch 5039/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2344 - binary_accuracy: 1.0000
    Epoch 5040/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2344 - binary_accuracy: 1.0000
    Epoch 5041/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2343 - binary_accuracy: 1.0000
    Epoch 5042/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2343 - binary_accuracy: 1.0000
    Epoch 5043/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2343 - binary_accuracy: 1.0000
    Epoch 5044/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2342 - binary_accuracy: 1.0000
    Epoch 5045/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2342 - binary_accuracy: 1.0000
    Epoch 5046/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2342 - binary_accuracy: 1.0000
    Epoch 5047/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2342 - binary_accuracy: 1.0000
    Epoch 5048/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2341 - binary_accuracy: 1.0000
    Epoch 5049/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2341 - binary_accuracy: 1.0000
    Epoch 5050/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2341 - binary_accuracy: 1.0000
    Epoch 5051/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2340 - binary_accuracy: 1.0000
    Epoch 5052/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2340 - binary_accuracy: 1.0000
    Epoch 5053/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2340 - binary_accuracy: 1.0000
    Epoch 5054/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2340 - binary_accuracy: 1.0000
    Epoch 5055/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2339 - binary_accuracy: 1.0000
    Epoch 5056/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2339 - binary_accuracy: 1.0000
    Epoch 5057/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2339 - binary_accuracy: 1.0000
    Epoch 5058/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2338 - binary_accuracy: 1.0000
    Epoch 5059/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2338 - binary_accuracy: 1.0000
    Epoch 5060/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2338 - binary_accuracy: 1.0000
    Epoch 5061/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2338 - binary_accuracy: 1.0000
    Epoch 5062/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2337 - binary_accuracy: 1.0000
    Epoch 5063/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2337 - binary_accuracy: 1.0000
    Epoch 5064/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2337 - binary_accuracy: 1.0000
    Epoch 5065/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2336 - binary_accuracy: 1.0000
    Epoch 5066/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2336 - binary_accuracy: 1.0000
    Epoch 5067/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2336 - binary_accuracy: 1.0000
    Epoch 5068/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2336 - binary_accuracy: 1.0000
    Epoch 5069/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2335 - binary_accuracy: 1.0000
    Epoch 5070/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2335 - binary_accuracy: 1.0000
    Epoch 5071/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2335 - binary_accuracy: 1.0000
    Epoch 5072/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2334 - binary_accuracy: 1.0000
    Epoch 5073/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2334 - binary_accuracy: 1.0000
    Epoch 5074/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2334 - binary_accuracy: 1.0000
    Epoch 5075/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2334 - binary_accuracy: 1.0000
    Epoch 5076/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2333 - binary_accuracy: 1.0000
    Epoch 5077/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2333 - binary_accuracy: 1.0000
    Epoch 5078/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2333 - binary_accuracy: 1.0000
    Epoch 5079/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2332 - binary_accuracy: 1.0000
    Epoch 5080/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2332 - binary_accuracy: 1.0000
    Epoch 5081/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2332 - binary_accuracy: 1.0000
    Epoch 5082/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2332 - binary_accuracy: 1.0000
    Epoch 5083/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2331 - binary_accuracy: 1.0000
    Epoch 5084/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2331 - binary_accuracy: 1.0000
    Epoch 5085/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2331 - binary_accuracy: 1.0000
    Epoch 5086/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2330 - binary_accuracy: 1.0000
    Epoch 5087/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2330 - binary_accuracy: 1.0000
    Epoch 5088/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2330 - binary_accuracy: 1.0000
    Epoch 5089/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2330 - binary_accuracy: 1.0000
    Epoch 5090/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2329 - binary_accuracy: 1.0000
    Epoch 5091/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2329 - binary_accuracy: 1.0000
    Epoch 5092/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2329 - binary_accuracy: 1.0000
    Epoch 5093/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2328 - binary_accuracy: 1.0000
    Epoch 5094/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2328 - binary_accuracy: 1.0000
    Epoch 5095/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2328 - binary_accuracy: 1.0000
    Epoch 5096/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2328 - binary_accuracy: 1.0000
    Epoch 5097/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2327 - binary_accuracy: 1.0000
    Epoch 5098/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2327 - binary_accuracy: 1.0000
    Epoch 5099/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2327 - binary_accuracy: 1.0000
    Epoch 5100/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2326 - binary_accuracy: 1.0000
    Epoch 5101/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2326 - binary_accuracy: 1.0000
    Epoch 5102/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2326 - binary_accuracy: 1.0000
    Epoch 5103/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2326 - binary_accuracy: 1.0000
    Epoch 5104/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2325 - binary_accuracy: 1.0000
    Epoch 5105/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2325 - binary_accuracy: 1.0000
    Epoch 5106/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2325 - binary_accuracy: 1.0000
    Epoch 5107/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2324 - binary_accuracy: 1.0000
    Epoch 5108/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2324 - binary_accuracy: 1.0000
    Epoch 5109/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2324 - binary_accuracy: 1.0000
    Epoch 5110/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2324 - binary_accuracy: 1.0000
    Epoch 5111/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2323 - binary_accuracy: 1.0000
    Epoch 5112/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2323 - binary_accuracy: 1.0000
    Epoch 5113/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2323 - binary_accuracy: 1.0000
    Epoch 5114/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2322 - binary_accuracy: 1.0000
    Epoch 5115/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2322 - binary_accuracy: 1.0000
    Epoch 5116/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2322 - binary_accuracy: 1.0000
    Epoch 5117/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2322 - binary_accuracy: 1.0000
    Epoch 5118/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2321 - binary_accuracy: 1.0000
    Epoch 5119/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2321 - binary_accuracy: 1.0000
    Epoch 5120/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2321 - binary_accuracy: 1.0000
    Epoch 5121/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2320 - binary_accuracy: 1.0000
    Epoch 5122/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2320 - binary_accuracy: 1.0000
    Epoch 5123/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2320 - binary_accuracy: 1.0000
    Epoch 5124/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2320 - binary_accuracy: 1.0000
    Epoch 5125/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2319 - binary_accuracy: 1.0000
    Epoch 5126/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2319 - binary_accuracy: 1.0000
    Epoch 5127/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2319 - binary_accuracy: 1.0000
    Epoch 5128/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2319 - binary_accuracy: 1.0000
    Epoch 5129/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2318 - binary_accuracy: 1.0000
    Epoch 5130/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2318 - binary_accuracy: 1.0000
    Epoch 5131/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2318 - binary_accuracy: 1.0000
    Epoch 5132/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2317 - binary_accuracy: 1.0000
    Epoch 5133/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2317 - binary_accuracy: 1.0000
    Epoch 5134/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2317 - binary_accuracy: 1.0000
    Epoch 5135/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2317 - binary_accuracy: 1.0000
    Epoch 5136/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2316 - binary_accuracy: 1.0000
    Epoch 5137/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2316 - binary_accuracy: 1.0000
    Epoch 5138/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2316 - binary_accuracy: 1.0000
    Epoch 5139/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2315 - binary_accuracy: 1.0000
    Epoch 5140/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2315 - binary_accuracy: 1.0000
    Epoch 5141/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2315 - binary_accuracy: 1.0000
    Epoch 5142/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2315 - binary_accuracy: 1.0000
    Epoch 5143/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2314 - binary_accuracy: 1.0000
    Epoch 5144/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2314 - binary_accuracy: 1.0000
    Epoch 5145/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2314 - binary_accuracy: 1.0000
    Epoch 5146/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2313 - binary_accuracy: 1.0000
    Epoch 5147/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2313 - binary_accuracy: 1.0000
    Epoch 5148/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2313 - binary_accuracy: 1.0000
    Epoch 5149/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2313 - binary_accuracy: 1.0000
    Epoch 5150/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2312 - binary_accuracy: 1.0000
    Epoch 5151/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2312 - binary_accuracy: 1.0000
    Epoch 5152/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2312 - binary_accuracy: 1.0000
    Epoch 5153/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2311 - binary_accuracy: 1.0000
    Epoch 5154/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2311 - binary_accuracy: 1.0000
    Epoch 5155/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2311 - binary_accuracy: 1.0000
    Epoch 5156/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2311 - binary_accuracy: 1.0000
    Epoch 5157/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2310 - binary_accuracy: 1.0000
    Epoch 5158/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2310 - binary_accuracy: 1.0000
    Epoch 5159/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2310 - binary_accuracy: 1.0000
    Epoch 5160/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2310 - binary_accuracy: 1.0000
    Epoch 5161/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2309 - binary_accuracy: 1.0000
    Epoch 5162/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2309 - binary_accuracy: 1.0000
    Epoch 5163/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2309 - binary_accuracy: 1.0000
    Epoch 5164/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2308 - binary_accuracy: 1.0000
    Epoch 5165/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2308 - binary_accuracy: 1.0000
    Epoch 5166/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2308 - binary_accuracy: 1.0000
    Epoch 5167/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2308 - binary_accuracy: 1.0000
    Epoch 5168/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2307 - binary_accuracy: 1.0000
    Epoch 5169/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2307 - binary_accuracy: 1.0000
    Epoch 5170/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2307 - binary_accuracy: 1.0000
    Epoch 5171/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2306 - binary_accuracy: 1.0000
    Epoch 5172/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2306 - binary_accuracy: 1.0000
    Epoch 5173/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2306 - binary_accuracy: 1.0000
    Epoch 5174/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2306 - binary_accuracy: 1.0000
    Epoch 5175/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2305 - binary_accuracy: 1.0000
    Epoch 5176/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2305 - binary_accuracy: 1.0000
    Epoch 5177/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2305 - binary_accuracy: 1.0000
    Epoch 5178/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2305 - binary_accuracy: 1.0000
    Epoch 5179/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2304 - binary_accuracy: 1.0000
    Epoch 5180/7000
    1/1 [==============================] - 0s 32ms/step - loss: 0.2304 - binary_accuracy: 1.0000
    Epoch 5181/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2304 - binary_accuracy: 1.0000
    Epoch 5182/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2303 - binary_accuracy: 1.0000
    Epoch 5183/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2303 - binary_accuracy: 1.0000
    Epoch 5184/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2303 - binary_accuracy: 1.0000
    Epoch 5185/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2303 - binary_accuracy: 1.0000
    Epoch 5186/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2302 - binary_accuracy: 1.0000
    Epoch 5187/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2302 - binary_accuracy: 1.0000
    Epoch 5188/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2302 - binary_accuracy: 1.0000
    Epoch 5189/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2301 - binary_accuracy: 1.0000
    Epoch 5190/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2301 - binary_accuracy: 1.0000
    Epoch 5191/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2301 - binary_accuracy: 1.0000
    Epoch 5192/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2301 - binary_accuracy: 1.0000
    Epoch 5193/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2300 - binary_accuracy: 1.0000
    Epoch 5194/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2300 - binary_accuracy: 1.0000
    Epoch 5195/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2300 - binary_accuracy: 1.0000
    Epoch 5196/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2300 - binary_accuracy: 1.0000
    Epoch 5197/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2299 - binary_accuracy: 1.0000
    Epoch 5198/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2299 - binary_accuracy: 1.0000
    Epoch 5199/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2299 - binary_accuracy: 1.0000
    Epoch 5200/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2298 - binary_accuracy: 1.0000
    Epoch 5201/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2298 - binary_accuracy: 1.0000
    Epoch 5202/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2298 - binary_accuracy: 1.0000
    Epoch 5203/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2298 - binary_accuracy: 1.0000
    Epoch 5204/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2297 - binary_accuracy: 1.0000
    Epoch 5205/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2297 - binary_accuracy: 1.0000
    Epoch 5206/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2297 - binary_accuracy: 1.0000
    Epoch 5207/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2296 - binary_accuracy: 1.0000
    Epoch 5208/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2296 - binary_accuracy: 1.0000
    Epoch 5209/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2296 - binary_accuracy: 1.0000
    Epoch 5210/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2296 - binary_accuracy: 1.0000
    Epoch 5211/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2295 - binary_accuracy: 1.0000
    Epoch 5212/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2295 - binary_accuracy: 1.0000
    Epoch 5213/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2295 - binary_accuracy: 1.0000
    Epoch 5214/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2295 - binary_accuracy: 1.0000
    Epoch 5215/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2294 - binary_accuracy: 1.0000
    Epoch 5216/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2294 - binary_accuracy: 1.0000
    Epoch 5217/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2294 - binary_accuracy: 1.0000
    Epoch 5218/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2293 - binary_accuracy: 1.0000
    Epoch 5219/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2293 - binary_accuracy: 1.0000
    Epoch 5220/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2293 - binary_accuracy: 1.0000
    Epoch 5221/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2293 - binary_accuracy: 1.0000
    Epoch 5222/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2292 - binary_accuracy: 1.0000
    Epoch 5223/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2292 - binary_accuracy: 1.0000
    Epoch 5224/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2292 - binary_accuracy: 1.0000
    Epoch 5225/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2292 - binary_accuracy: 1.0000
    Epoch 5226/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2291 - binary_accuracy: 1.0000
    Epoch 5227/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2291 - binary_accuracy: 1.0000
    Epoch 5228/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2291 - binary_accuracy: 1.0000
    Epoch 5229/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2290 - binary_accuracy: 1.0000
    Epoch 5230/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2290 - binary_accuracy: 1.0000
    Epoch 5231/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2290 - binary_accuracy: 1.0000
    Epoch 5232/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2290 - binary_accuracy: 1.0000
    Epoch 5233/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2289 - binary_accuracy: 1.0000
    Epoch 5234/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2289 - binary_accuracy: 1.0000
    Epoch 5235/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2289 - binary_accuracy: 1.0000
    Epoch 5236/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.2288 - binary_accuracy: 1.0000
    Epoch 5237/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2288 - binary_accuracy: 1.0000
    Epoch 5238/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2288 - binary_accuracy: 1.0000
    Epoch 5239/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2288 - binary_accuracy: 1.0000
    Epoch 5240/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2287 - binary_accuracy: 1.0000
    Epoch 5241/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2287 - binary_accuracy: 1.0000
    Epoch 5242/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2287 - binary_accuracy: 1.0000
    Epoch 5243/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2287 - binary_accuracy: 1.0000
    Epoch 5244/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2286 - binary_accuracy: 1.0000
    Epoch 5245/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2286 - binary_accuracy: 1.0000
    Epoch 5246/7000
    1/1 [==============================] - 0s 15ms/step - loss: 0.2286 - binary_accuracy: 1.0000
    Epoch 5247/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2285 - binary_accuracy: 1.0000
    Epoch 5248/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2285 - binary_accuracy: 1.0000
    Epoch 5249/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2285 - binary_accuracy: 1.0000
    Epoch 5250/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2285 - binary_accuracy: 1.0000
    Epoch 5251/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2284 - binary_accuracy: 1.0000
    Epoch 5252/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2284 - binary_accuracy: 1.0000
    Epoch 5253/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2284 - binary_accuracy: 1.0000
    Epoch 5254/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2284 - binary_accuracy: 1.0000
    Epoch 5255/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2283 - binary_accuracy: 1.0000
    Epoch 5256/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2283 - binary_accuracy: 1.0000
    Epoch 5257/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2283 - binary_accuracy: 1.0000
    Epoch 5258/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2282 - binary_accuracy: 1.0000
    Epoch 5259/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2282 - binary_accuracy: 1.0000
    Epoch 5260/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2282 - binary_accuracy: 1.0000
    Epoch 5261/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2282 - binary_accuracy: 1.0000
    Epoch 5262/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2281 - binary_accuracy: 1.0000
    Epoch 5263/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2281 - binary_accuracy: 1.0000
    Epoch 5264/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2281 - binary_accuracy: 1.0000
    Epoch 5265/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2281 - binary_accuracy: 1.0000
    Epoch 5266/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2280 - binary_accuracy: 1.0000
    Epoch 5267/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2280 - binary_accuracy: 1.0000
    Epoch 5268/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2280 - binary_accuracy: 1.0000
    Epoch 5269/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2279 - binary_accuracy: 1.0000
    Epoch 5270/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2279 - binary_accuracy: 1.0000
    Epoch 5271/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2279 - binary_accuracy: 1.0000
    Epoch 5272/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2279 - binary_accuracy: 1.0000
    Epoch 5273/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2278 - binary_accuracy: 1.0000
    Epoch 5274/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2278 - binary_accuracy: 1.0000
    Epoch 5275/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2278 - binary_accuracy: 1.0000
    Epoch 5276/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2278 - binary_accuracy: 1.0000
    Epoch 5277/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2277 - binary_accuracy: 1.0000
    Epoch 5278/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2277 - binary_accuracy: 1.0000
    Epoch 5279/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2277 - binary_accuracy: 1.0000
    Epoch 5280/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2276 - binary_accuracy: 1.0000
    Epoch 5281/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2276 - binary_accuracy: 1.0000
    Epoch 5282/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2276 - binary_accuracy: 1.0000
    Epoch 5283/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2276 - binary_accuracy: 1.0000
    Epoch 5284/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2275 - binary_accuracy: 1.0000
    Epoch 5285/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.2275 - binary_accuracy: 1.0000
    Epoch 5286/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2275 - binary_accuracy: 1.0000
    Epoch 5287/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2275 - binary_accuracy: 1.0000
    Epoch 5288/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2274 - binary_accuracy: 1.0000
    Epoch 5289/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2274 - binary_accuracy: 1.0000
    Epoch 5290/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2274 - binary_accuracy: 1.0000
    Epoch 5291/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2274 - binary_accuracy: 1.0000
    Epoch 5292/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2273 - binary_accuracy: 1.0000
    Epoch 5293/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2273 - binary_accuracy: 1.0000
    Epoch 5294/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2273 - binary_accuracy: 1.0000
    Epoch 5295/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2272 - binary_accuracy: 1.0000
    Epoch 5296/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2272 - binary_accuracy: 1.0000
    Epoch 5297/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2272 - binary_accuracy: 1.0000
    Epoch 5298/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2272 - binary_accuracy: 1.0000
    Epoch 5299/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2271 - binary_accuracy: 1.0000
    Epoch 5300/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2271 - binary_accuracy: 1.0000
    Epoch 5301/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2271 - binary_accuracy: 1.0000
    Epoch 5302/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2271 - binary_accuracy: 1.0000
    Epoch 5303/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2270 - binary_accuracy: 1.0000
    Epoch 5304/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2270 - binary_accuracy: 1.0000
    Epoch 5305/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2270 - binary_accuracy: 1.0000
    Epoch 5306/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2269 - binary_accuracy: 1.0000
    Epoch 5307/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2269 - binary_accuracy: 1.0000
    Epoch 5308/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2269 - binary_accuracy: 1.0000
    Epoch 5309/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2269 - binary_accuracy: 1.0000
    Epoch 5310/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2268 - binary_accuracy: 1.0000
    Epoch 5311/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2268 - binary_accuracy: 1.0000
    Epoch 5312/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2268 - binary_accuracy: 1.0000
    Epoch 5313/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2268 - binary_accuracy: 1.0000
    Epoch 5314/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2267 - binary_accuracy: 1.0000
    Epoch 5315/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2267 - binary_accuracy: 1.0000
    Epoch 5316/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.2267 - binary_accuracy: 1.0000
    Epoch 5317/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2266 - binary_accuracy: 1.0000
    Epoch 5318/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2266 - binary_accuracy: 1.0000
    Epoch 5319/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2266 - binary_accuracy: 1.0000
    Epoch 5320/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2266 - binary_accuracy: 1.0000
    Epoch 5321/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2265 - binary_accuracy: 1.0000
    Epoch 5322/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2265 - binary_accuracy: 1.0000
    Epoch 5323/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2265 - binary_accuracy: 1.0000
    Epoch 5324/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2265 - binary_accuracy: 1.0000
    Epoch 5325/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2264 - binary_accuracy: 1.0000
    Epoch 5326/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2264 - binary_accuracy: 1.0000
    Epoch 5327/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2264 - binary_accuracy: 1.0000
    Epoch 5328/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2131 - binary_accuracy: 1.0000
    Epoch 5857/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2130 - binary_accuracy: 1.0000
    Epoch 5858/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2130 - binary_accuracy: 1.0000
    Epoch 5859/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2130 - binary_accuracy: 1.0000
    Epoch 5860/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2130 - binary_accuracy: 1.0000
    Epoch 5861/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2129 - binary_accuracy: 1.0000
    Epoch 5862/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2129 - binary_accuracy: 1.0000
    Epoch 5863/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2129 - binary_accuracy: 1.0000
    Epoch 5864/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2129 - binary_accuracy: 1.0000
    Epoch 5865/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2129 - binary_accuracy: 1.0000
    Epoch 5866/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2128 - binary_accuracy: 1.0000
    Epoch 5867/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2128 - binary_accuracy: 1.0000
    Epoch 5868/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2128 - binary_accuracy: 1.0000
    Epoch 5869/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2128 - binary_accuracy: 1.0000
    Epoch 5870/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2127 - binary_accuracy: 1.0000
    Epoch 5871/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2127 - binary_accuracy: 1.0000
    Epoch 5872/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2127 - binary_accuracy: 1.0000
    Epoch 5873/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2127 - binary_accuracy: 1.0000
    Epoch 5874/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2126 - binary_accuracy: 1.0000
    Epoch 5875/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2126 - binary_accuracy: 1.0000
    Epoch 5876/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2126 - binary_accuracy: 1.0000
    Epoch 5877/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2126 - binary_accuracy: 1.0000
    Epoch 5878/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2125 - binary_accuracy: 1.0000
    Epoch 5879/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2125 - binary_accuracy: 1.0000
    Epoch 5880/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2125 - binary_accuracy: 1.0000
    Epoch 5881/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2125 - binary_accuracy: 1.0000
    Epoch 5882/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2125 - binary_accuracy: 1.0000
    Epoch 5883/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2124 - binary_accuracy: 1.0000
    Epoch 5884/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2124 - binary_accuracy: 1.0000
    Epoch 5885/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2124 - binary_accuracy: 1.0000
    Epoch 5886/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2124 - binary_accuracy: 1.0000
    Epoch 5887/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2123 - binary_accuracy: 1.0000
    Epoch 5888/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2123 - binary_accuracy: 1.0000
    Epoch 5889/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2123 - binary_accuracy: 1.0000
    Epoch 5890/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2123 - binary_accuracy: 1.0000
    Epoch 5891/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2122 - binary_accuracy: 1.0000
    Epoch 5892/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2122 - binary_accuracy: 1.0000
    Epoch 5893/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2122 - binary_accuracy: 1.0000
    Epoch 5894/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2122 - binary_accuracy: 1.0000
    Epoch 5895/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2122 - binary_accuracy: 1.0000
    Epoch 5896/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2121 - binary_accuracy: 1.0000
    Epoch 5897/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2121 - binary_accuracy: 1.0000
    Epoch 5898/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2121 - binary_accuracy: 1.0000
    Epoch 5899/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2121 - binary_accuracy: 1.0000
    Epoch 5900/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2120 - binary_accuracy: 1.0000
    Epoch 5901/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2120 - binary_accuracy: 1.0000
    Epoch 5902/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2120 - binary_accuracy: 1.0000
    Epoch 5903/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2120 - binary_accuracy: 1.0000
    Epoch 5904/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2119 - binary_accuracy: 1.0000
    Epoch 5905/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2119 - binary_accuracy: 1.0000
    Epoch 5906/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2119 - binary_accuracy: 1.0000
    Epoch 5907/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2119 - binary_accuracy: 1.0000
    Epoch 5908/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2118 - binary_accuracy: 1.0000
    Epoch 5909/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2118 - binary_accuracy: 1.0000
    Epoch 5910/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2118 - binary_accuracy: 1.0000
    Epoch 5911/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2118 - binary_accuracy: 1.0000
    Epoch 5912/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2118 - binary_accuracy: 1.0000
    Epoch 5913/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2117 - binary_accuracy: 1.0000
    Epoch 5914/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2117 - binary_accuracy: 1.0000
    Epoch 5915/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2117 - binary_accuracy: 1.0000
    Epoch 5916/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2117 - binary_accuracy: 1.0000
    Epoch 5917/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2116 - binary_accuracy: 1.0000
    Epoch 5918/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2116 - binary_accuracy: 1.0000
    Epoch 5919/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2116 - binary_accuracy: 1.0000
    Epoch 5920/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2116 - binary_accuracy: 1.0000
    Epoch 5921/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2115 - binary_accuracy: 1.0000
    Epoch 5922/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2115 - binary_accuracy: 1.0000
    Epoch 5923/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2115 - binary_accuracy: 1.0000
    Epoch 5924/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2115 - binary_accuracy: 1.0000
    Epoch 5925/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2115 - binary_accuracy: 1.0000
    Epoch 5926/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2114 - binary_accuracy: 1.0000
    Epoch 5927/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2114 - binary_accuracy: 1.0000
    Epoch 5928/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2114 - binary_accuracy: 1.0000
    Epoch 5929/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2114 - binary_accuracy: 1.0000
    Epoch 5930/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2113 - binary_accuracy: 1.0000
    Epoch 5931/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2113 - binary_accuracy: 1.0000
    Epoch 5932/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2113 - binary_accuracy: 1.0000
    Epoch 5933/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2113 - binary_accuracy: 1.0000
    Epoch 5934/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2112 - binary_accuracy: 1.0000
    Epoch 5935/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2112 - binary_accuracy: 1.0000
    Epoch 5936/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2112 - binary_accuracy: 1.0000
    Epoch 5937/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2112 - binary_accuracy: 1.0000
    Epoch 5938/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2111 - binary_accuracy: 1.0000
    Epoch 5939/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2111 - binary_accuracy: 1.0000
    Epoch 5940/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2111 - binary_accuracy: 1.0000
    Epoch 5941/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2111 - binary_accuracy: 1.0000
    Epoch 5942/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2111 - binary_accuracy: 1.0000
    Epoch 5943/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2110 - binary_accuracy: 1.0000
    Epoch 5944/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2110 - binary_accuracy: 1.0000
    Epoch 5945/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2110 - binary_accuracy: 1.0000
    Epoch 5946/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2110 - binary_accuracy: 1.0000
    Epoch 5947/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2109 - binary_accuracy: 1.0000
    Epoch 5948/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2109 - binary_accuracy: 1.0000
    Epoch 5949/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2109 - binary_accuracy: 1.0000
    Epoch 5950/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2109 - binary_accuracy: 1.0000
    Epoch 5951/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2108 - binary_accuracy: 1.0000
    Epoch 5952/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2108 - binary_accuracy: 1.0000
    Epoch 5953/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2108 - binary_accuracy: 1.0000
    Epoch 5954/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2108 - binary_accuracy: 1.0000
    Epoch 5955/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2108 - binary_accuracy: 1.0000
    Epoch 5956/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2107 - binary_accuracy: 1.0000
    Epoch 5957/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2107 - binary_accuracy: 1.0000
    Epoch 5958/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2107 - binary_accuracy: 1.0000
    Epoch 5959/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2107 - binary_accuracy: 1.0000
    Epoch 5960/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2106 - binary_accuracy: 1.0000
    Epoch 5961/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2106 - binary_accuracy: 1.0000
    Epoch 5962/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2106 - binary_accuracy: 1.0000
    Epoch 5963/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2106 - binary_accuracy: 1.0000
    Epoch 5964/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2105 - binary_accuracy: 1.0000
    Epoch 5965/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2105 - binary_accuracy: 1.0000
    Epoch 5966/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2105 - binary_accuracy: 1.0000
    Epoch 5967/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2105 - binary_accuracy: 1.0000
    Epoch 5968/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2105 - binary_accuracy: 1.0000
    Epoch 5969/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2104 - binary_accuracy: 1.0000
    Epoch 5970/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2104 - binary_accuracy: 1.0000
    Epoch 5971/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2104 - binary_accuracy: 1.0000
    Epoch 5972/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2104 - binary_accuracy: 1.0000
    Epoch 5973/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2103 - binary_accuracy: 1.0000
    Epoch 5974/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2103 - binary_accuracy: 1.0000
    Epoch 5975/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2103 - binary_accuracy: 1.0000
    Epoch 5976/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2103 - binary_accuracy: 1.0000
    Epoch 5977/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2102 - binary_accuracy: 1.0000
    Epoch 5978/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2102 - binary_accuracy: 1.0000
    Epoch 5979/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2102 - binary_accuracy: 1.0000
    Epoch 5980/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2102 - binary_accuracy: 1.0000
    Epoch 5981/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2102 - binary_accuracy: 1.0000
    Epoch 5982/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2101 - binary_accuracy: 1.0000
    Epoch 5983/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2101 - binary_accuracy: 1.0000
    Epoch 5984/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2101 - binary_accuracy: 1.0000
    Epoch 5985/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2101 - binary_accuracy: 1.0000
    Epoch 5986/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2100 - binary_accuracy: 1.0000
    Epoch 5987/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2100 - binary_accuracy: 1.0000
    Epoch 5988/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2100 - binary_accuracy: 1.0000
    Epoch 5989/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2100 - binary_accuracy: 1.0000
    Epoch 5990/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2099 - binary_accuracy: 1.0000
    Epoch 5991/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2099 - binary_accuracy: 1.0000
    Epoch 5992/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2099 - binary_accuracy: 1.0000
    Epoch 5993/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2099 - binary_accuracy: 1.0000
    Epoch 5994/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2099 - binary_accuracy: 1.0000
    Epoch 5995/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2098 - binary_accuracy: 1.0000
    Epoch 5996/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2098 - binary_accuracy: 1.0000
    Epoch 5997/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2098 - binary_accuracy: 1.0000
    Epoch 5998/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2098 - binary_accuracy: 1.0000
    Epoch 5999/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2097 - binary_accuracy: 1.0000
    Epoch 6000/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2097 - binary_accuracy: 1.0000
    Epoch 6001/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2097 - binary_accuracy: 1.0000
    Epoch 6002/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2097 - binary_accuracy: 1.0000
    Epoch 6003/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2097 - binary_accuracy: 1.0000
    Epoch 6004/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2096 - binary_accuracy: 1.0000
    Epoch 6005/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2096 - binary_accuracy: 1.0000
    Epoch 6006/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2096 - binary_accuracy: 1.0000
    Epoch 6007/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2096 - binary_accuracy: 1.0000
    Epoch 6008/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2095 - binary_accuracy: 1.0000
    Epoch 6009/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2095 - binary_accuracy: 1.0000
    Epoch 6010/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2095 - binary_accuracy: 1.0000
    Epoch 6011/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2095 - binary_accuracy: 1.0000
    Epoch 6012/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2094 - binary_accuracy: 1.0000
    Epoch 6013/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2094 - binary_accuracy: 1.0000
    Epoch 6014/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2094 - binary_accuracy: 1.0000
    Epoch 6015/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2094 - binary_accuracy: 1.0000
    Epoch 6016/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2094 - binary_accuracy: 1.0000
    Epoch 6017/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2093 - binary_accuracy: 1.0000
    Epoch 6018/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2093 - binary_accuracy: 1.0000
    Epoch 6019/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2093 - binary_accuracy: 1.0000
    Epoch 6020/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2093 - binary_accuracy: 1.0000
    Epoch 6021/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2092 - binary_accuracy: 1.0000
    Epoch 6022/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2092 - binary_accuracy: 1.0000
    Epoch 6023/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2092 - binary_accuracy: 1.0000
    Epoch 6024/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2092 - binary_accuracy: 1.0000
    Epoch 6025/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2092 - binary_accuracy: 1.0000
    Epoch 6026/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2091 - binary_accuracy: 1.0000
    Epoch 6027/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2091 - binary_accuracy: 1.0000
    Epoch 6028/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2091 - binary_accuracy: 1.0000
    Epoch 6029/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2091 - binary_accuracy: 1.0000
    Epoch 6030/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2090 - binary_accuracy: 1.0000
    Epoch 6031/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2090 - binary_accuracy: 1.0000
    Epoch 6032/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2090 - binary_accuracy: 1.0000
    Epoch 6033/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2090 - binary_accuracy: 1.0000
    Epoch 6034/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2089 - binary_accuracy: 1.0000
    Epoch 6035/7000
    1/1 [==============================] - 0s 48ms/step - loss: 0.2089 - binary_accuracy: 1.0000
    Epoch 6036/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2089 - binary_accuracy: 1.0000
    Epoch 6037/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2089 - binary_accuracy: 1.0000
    Epoch 6038/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2089 - binary_accuracy: 1.0000
    Epoch 6039/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2088 - binary_accuracy: 1.0000
    Epoch 6040/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2088 - binary_accuracy: 1.0000
    Epoch 6041/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2088 - binary_accuracy: 1.0000
    Epoch 6042/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2088 - binary_accuracy: 1.0000
    Epoch 6043/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2087 - binary_accuracy: 1.0000
    Epoch 6044/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2087 - binary_accuracy: 1.0000
    Epoch 6045/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2087 - binary_accuracy: 1.0000
    Epoch 6046/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2087 - binary_accuracy: 1.0000
    Epoch 6047/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2087 - binary_accuracy: 1.0000
    Epoch 6048/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2086 - binary_accuracy: 1.0000
    Epoch 6049/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2086 - binary_accuracy: 1.0000
    Epoch 6050/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2086 - binary_accuracy: 1.0000
    Epoch 6051/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2086 - binary_accuracy: 1.0000
    Epoch 6052/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2085 - binary_accuracy: 1.0000
    Epoch 6053/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2085 - binary_accuracy: 1.0000
    Epoch 6054/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2085 - binary_accuracy: 1.0000
    Epoch 6055/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2085 - binary_accuracy: 1.0000
    Epoch 6056/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2084 - binary_accuracy: 1.0000
    Epoch 6057/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2084 - binary_accuracy: 1.0000
    Epoch 6058/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2084 - binary_accuracy: 1.0000
    Epoch 6059/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2084 - binary_accuracy: 1.0000
    Epoch 6060/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2084 - binary_accuracy: 1.0000
    Epoch 6061/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2083 - binary_accuracy: 1.0000
    Epoch 6062/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2083 - binary_accuracy: 1.0000
    Epoch 6063/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2083 - binary_accuracy: 1.0000
    Epoch 6064/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2083 - binary_accuracy: 1.0000
    Epoch 6065/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2082 - binary_accuracy: 1.0000
    Epoch 6066/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2082 - binary_accuracy: 1.0000
    Epoch 6067/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2082 - binary_accuracy: 1.0000
    Epoch 6068/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2082 - binary_accuracy: 1.0000
    Epoch 6069/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2082 - binary_accuracy: 1.0000
    Epoch 6070/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2081 - binary_accuracy: 1.0000
    Epoch 6071/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2081 - binary_accuracy: 1.0000
    Epoch 6072/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2081 - binary_accuracy: 1.0000
    Epoch 6073/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2081 - binary_accuracy: 1.0000
    Epoch 6074/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2080 - binary_accuracy: 1.0000
    Epoch 6075/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2080 - binary_accuracy: 1.0000
    Epoch 6076/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2080 - binary_accuracy: 1.0000
    Epoch 6077/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2080 - binary_accuracy: 1.0000
    Epoch 6078/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2080 - binary_accuracy: 1.0000
    Epoch 6079/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2079 - binary_accuracy: 1.0000
    Epoch 6080/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2079 - binary_accuracy: 1.0000
    Epoch 6081/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2079 - binary_accuracy: 1.0000
    Epoch 6082/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2079 - binary_accuracy: 1.0000
    Epoch 6083/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2078 - binary_accuracy: 1.0000
    Epoch 6084/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2078 - binary_accuracy: 1.0000
    Epoch 6085/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2078 - binary_accuracy: 1.0000
    Epoch 6086/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2078 - binary_accuracy: 1.0000
    Epoch 6087/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2078 - binary_accuracy: 1.0000
    Epoch 6088/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2077 - binary_accuracy: 1.0000
    Epoch 6089/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2077 - binary_accuracy: 1.0000
    Epoch 6090/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2077 - binary_accuracy: 1.0000
    Epoch 6091/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2077 - binary_accuracy: 1.0000
    Epoch 6092/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2076 - binary_accuracy: 1.0000
    Epoch 6093/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2076 - binary_accuracy: 1.0000
    Epoch 6094/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2076 - binary_accuracy: 1.0000
    Epoch 6095/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2076 - binary_accuracy: 1.0000
    Epoch 6096/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2075 - binary_accuracy: 1.0000
    Epoch 6097/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2075 - binary_accuracy: 1.0000
    Epoch 6098/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2075 - binary_accuracy: 1.0000
    Epoch 6099/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2075 - binary_accuracy: 1.0000
    Epoch 6100/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2075 - binary_accuracy: 1.0000
    Epoch 6101/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2074 - binary_accuracy: 1.0000
    Epoch 6102/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2074 - binary_accuracy: 1.0000
    Epoch 6103/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2074 - binary_accuracy: 1.0000
    Epoch 6104/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2074 - binary_accuracy: 1.0000
    Epoch 6105/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2073 - binary_accuracy: 1.0000
    Epoch 6106/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2073 - binary_accuracy: 1.0000
    Epoch 6107/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2073 - binary_accuracy: 1.0000
    Epoch 6108/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2073 - binary_accuracy: 1.0000
    Epoch 6109/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2073 - binary_accuracy: 1.0000
    Epoch 6110/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2072 - binary_accuracy: 1.0000
    Epoch 6111/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2072 - binary_accuracy: 1.0000
    Epoch 6112/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2072 - binary_accuracy: 1.0000
    Epoch 6113/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2072 - binary_accuracy: 1.0000
    Epoch 6114/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2071 - binary_accuracy: 1.0000
    Epoch 6115/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2071 - binary_accuracy: 1.0000
    Epoch 6116/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2071 - binary_accuracy: 1.0000
    Epoch 6117/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2071 - binary_accuracy: 1.0000
    Epoch 6118/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2071 - binary_accuracy: 1.0000
    Epoch 6119/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2070 - binary_accuracy: 1.0000
    Epoch 6120/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2070 - binary_accuracy: 1.0000
    Epoch 6121/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2070 - binary_accuracy: 1.0000
    Epoch 6122/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2070 - binary_accuracy: 1.0000
    Epoch 6123/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2069 - binary_accuracy: 1.0000
    Epoch 6124/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2069 - binary_accuracy: 1.0000
    Epoch 6125/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2069 - binary_accuracy: 1.0000
    Epoch 6126/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2069 - binary_accuracy: 1.0000
    Epoch 6127/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2069 - binary_accuracy: 1.0000
    Epoch 6128/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2068 - binary_accuracy: 1.0000
    Epoch 6129/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2068 - binary_accuracy: 1.0000
    Epoch 6130/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2068 - binary_accuracy: 1.0000
    Epoch 6131/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2068 - binary_accuracy: 1.0000
    Epoch 6132/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2067 - binary_accuracy: 1.0000
    Epoch 6133/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2067 - binary_accuracy: 1.0000
    Epoch 6134/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2067 - binary_accuracy: 1.0000
    Epoch 6135/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2067 - binary_accuracy: 1.0000
    Epoch 6136/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2067 - binary_accuracy: 1.0000
    Epoch 6137/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2066 - binary_accuracy: 1.0000
    Epoch 6138/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2066 - binary_accuracy: 1.0000
    Epoch 6139/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2066 - binary_accuracy: 1.0000
    Epoch 6140/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2066 - binary_accuracy: 1.0000
    Epoch 6141/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2065 - binary_accuracy: 1.0000
    Epoch 6142/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2065 - binary_accuracy: 1.0000
    Epoch 6143/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2065 - binary_accuracy: 1.0000
    Epoch 6144/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2065 - binary_accuracy: 1.0000
    Epoch 6145/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2065 - binary_accuracy: 1.0000
    Epoch 6146/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2064 - binary_accuracy: 1.0000
    Epoch 6147/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2064 - binary_accuracy: 1.0000
    Epoch 6148/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2064 - binary_accuracy: 1.0000
    Epoch 6149/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2064 - binary_accuracy: 1.0000
    Epoch 6150/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2063 - binary_accuracy: 1.0000
    Epoch 6151/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2063 - binary_accuracy: 1.0000
    Epoch 6152/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2063 - binary_accuracy: 1.0000
    Epoch 6153/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2063 - binary_accuracy: 1.0000
    Epoch 6154/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2063 - binary_accuracy: 1.0000
    Epoch 6155/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2062 - binary_accuracy: 1.0000
    Epoch 6156/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2062 - binary_accuracy: 1.0000
    Epoch 6157/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2062 - binary_accuracy: 1.0000
    Epoch 6158/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2062 - binary_accuracy: 1.0000
    Epoch 6159/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2061 - binary_accuracy: 1.0000
    Epoch 6160/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2061 - binary_accuracy: 1.0000
    Epoch 6161/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2061 - binary_accuracy: 1.0000
    Epoch 6162/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2061 - binary_accuracy: 1.0000
    Epoch 6163/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2061 - binary_accuracy: 1.0000
    Epoch 6164/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2060 - binary_accuracy: 1.0000
    Epoch 6165/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2060 - binary_accuracy: 1.0000
    Epoch 6166/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2060 - binary_accuracy: 1.0000
    Epoch 6167/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2060 - binary_accuracy: 1.0000
    Epoch 6168/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2059 - binary_accuracy: 1.0000
    Epoch 6169/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2059 - binary_accuracy: 1.0000
    Epoch 6170/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2059 - binary_accuracy: 1.0000
    Epoch 6171/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2059 - binary_accuracy: 1.0000
    Epoch 6172/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2059 - binary_accuracy: 1.0000
    Epoch 6173/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2058 - binary_accuracy: 1.0000
    Epoch 6174/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2058 - binary_accuracy: 1.0000
    Epoch 6175/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2058 - binary_accuracy: 1.0000
    Epoch 6176/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2058 - binary_accuracy: 1.0000
    Epoch 6177/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2058 - binary_accuracy: 1.0000
    Epoch 6178/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2057 - binary_accuracy: 1.0000
    Epoch 6179/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2057 - binary_accuracy: 1.0000
    Epoch 6180/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2057 - binary_accuracy: 1.0000
    Epoch 6181/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2057 - binary_accuracy: 1.0000
    Epoch 6182/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2056 - binary_accuracy: 1.0000
    Epoch 6183/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2056 - binary_accuracy: 1.0000
    Epoch 6184/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2056 - binary_accuracy: 1.0000
    Epoch 6185/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2056 - binary_accuracy: 1.0000
    Epoch 6186/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2056 - binary_accuracy: 1.0000
    Epoch 6187/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2055 - binary_accuracy: 1.0000
    Epoch 6188/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2055 - binary_accuracy: 1.0000
    Epoch 6189/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2055 - binary_accuracy: 1.0000
    Epoch 6190/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2055 - binary_accuracy: 1.0000
    Epoch 6191/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2054 - binary_accuracy: 1.0000
    Epoch 6192/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2054 - binary_accuracy: 1.0000
    Epoch 6193/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2054 - binary_accuracy: 1.0000
    Epoch 6194/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2054 - binary_accuracy: 1.0000
    Epoch 6195/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2054 - binary_accuracy: 1.0000
    Epoch 6196/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2053 - binary_accuracy: 1.0000
    Epoch 6197/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2053 - binary_accuracy: 1.0000
    Epoch 6198/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2053 - binary_accuracy: 1.0000
    Epoch 6199/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2053 - binary_accuracy: 1.0000
    Epoch 6200/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2052 - binary_accuracy: 1.0000
    Epoch 6201/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2052 - binary_accuracy: 1.0000
    Epoch 6202/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2052 - binary_accuracy: 1.0000
    Epoch 6203/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2052 - binary_accuracy: 1.0000
    Epoch 6204/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2052 - binary_accuracy: 1.0000
    Epoch 6205/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2051 - binary_accuracy: 1.0000
    Epoch 6206/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2051 - binary_accuracy: 1.0000
    Epoch 6207/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2051 - binary_accuracy: 1.0000
    Epoch 6208/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2051 - binary_accuracy: 1.0000
    Epoch 6209/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2051 - binary_accuracy: 1.0000
    Epoch 6210/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2050 - binary_accuracy: 1.0000
    Epoch 6211/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2050 - binary_accuracy: 1.0000
    Epoch 6212/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2050 - binary_accuracy: 1.0000
    Epoch 6213/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2050 - binary_accuracy: 1.0000
    Epoch 6214/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2049 - binary_accuracy: 1.0000
    Epoch 6215/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2049 - binary_accuracy: 1.0000
    Epoch 6216/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2049 - binary_accuracy: 1.0000
    Epoch 6217/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2049 - binary_accuracy: 1.0000
    Epoch 6218/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2049 - binary_accuracy: 1.0000
    Epoch 6219/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2048 - binary_accuracy: 1.0000
    Epoch 6220/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2048 - binary_accuracy: 1.0000
    Epoch 6221/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2048 - binary_accuracy: 1.0000
    Epoch 6222/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2048 - binary_accuracy: 1.0000
    Epoch 6223/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2047 - binary_accuracy: 1.0000
    Epoch 6224/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2047 - binary_accuracy: 1.0000
    Epoch 6225/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2047 - binary_accuracy: 1.0000
    Epoch 6226/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2047 - binary_accuracy: 1.0000
    Epoch 6227/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2047 - binary_accuracy: 1.0000
    Epoch 6228/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2046 - binary_accuracy: 1.0000
    Epoch 6229/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2046 - binary_accuracy: 1.0000
    Epoch 6230/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2046 - binary_accuracy: 1.0000
    Epoch 6231/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2046 - binary_accuracy: 1.0000
    Epoch 6232/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2045 - binary_accuracy: 1.0000
    Epoch 6233/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2045 - binary_accuracy: 1.0000
    Epoch 6234/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2045 - binary_accuracy: 1.0000
    Epoch 6235/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2045 - binary_accuracy: 1.0000
    Epoch 6236/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2045 - binary_accuracy: 1.0000
    Epoch 6237/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2044 - binary_accuracy: 1.0000
    Epoch 6238/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2044 - binary_accuracy: 1.0000
    Epoch 6239/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2044 - binary_accuracy: 1.0000
    Epoch 6240/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2044 - binary_accuracy: 1.0000
    Epoch 6241/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2044 - binary_accuracy: 1.0000
    Epoch 6242/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2043 - binary_accuracy: 1.0000
    Epoch 6243/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2043 - binary_accuracy: 1.0000
    Epoch 6244/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2043 - binary_accuracy: 1.0000
    Epoch 6245/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2043 - binary_accuracy: 1.0000
    Epoch 6246/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2042 - binary_accuracy: 1.0000
    Epoch 6247/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2042 - binary_accuracy: 1.0000
    Epoch 6248/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2042 - binary_accuracy: 1.0000
    Epoch 6249/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2042 - binary_accuracy: 1.0000
    Epoch 6250/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2042 - binary_accuracy: 1.0000
    Epoch 6251/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2041 - binary_accuracy: 1.0000
    Epoch 6252/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2041 - binary_accuracy: 1.0000
    Epoch 6253/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2041 - binary_accuracy: 1.0000
    Epoch 6254/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2041 - binary_accuracy: 1.0000
    Epoch 6255/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2041 - binary_accuracy: 1.0000
    Epoch 6256/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2040 - binary_accuracy: 1.0000
    Epoch 6257/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2040 - binary_accuracy: 1.0000
    Epoch 6258/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2040 - binary_accuracy: 1.0000
    Epoch 6259/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2040 - binary_accuracy: 1.0000
    Epoch 6260/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2039 - binary_accuracy: 1.0000
    Epoch 6261/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2039 - binary_accuracy: 1.0000
    Epoch 6262/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2039 - binary_accuracy: 1.0000
    Epoch 6263/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2039 - binary_accuracy: 1.0000
    Epoch 6264/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2039 - binary_accuracy: 1.0000
    Epoch 6265/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2038 - binary_accuracy: 1.0000
    Epoch 6266/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2038 - binary_accuracy: 1.0000
    Epoch 6267/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2038 - binary_accuracy: 1.0000
    Epoch 6268/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2038 - binary_accuracy: 1.0000
    Epoch 6269/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2037 - binary_accuracy: 1.0000
    Epoch 6270/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2037 - binary_accuracy: 1.0000
    Epoch 6271/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2037 - binary_accuracy: 1.0000
    Epoch 6272/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2037 - binary_accuracy: 1.0000
    Epoch 6273/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2037 - binary_accuracy: 1.0000
    Epoch 6274/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2036 - binary_accuracy: 1.0000
    Epoch 6275/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2036 - binary_accuracy: 1.0000
    Epoch 6276/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2036 - binary_accuracy: 1.0000
    Epoch 6277/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2036 - binary_accuracy: 1.0000
    Epoch 6278/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2036 - binary_accuracy: 1.0000
    Epoch 6279/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2035 - binary_accuracy: 1.0000
    Epoch 6280/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2035 - binary_accuracy: 1.0000
    Epoch 6281/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2035 - binary_accuracy: 1.0000
    Epoch 6282/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2035 - binary_accuracy: 1.0000
    Epoch 6283/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2034 - binary_accuracy: 1.0000
    Epoch 6284/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2034 - binary_accuracy: 1.0000
    Epoch 6285/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2034 - binary_accuracy: 1.0000
    Epoch 6286/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2034 - binary_accuracy: 1.0000
    Epoch 6287/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2034 - binary_accuracy: 1.0000
    Epoch 6288/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2033 - binary_accuracy: 1.0000
    Epoch 6289/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2033 - binary_accuracy: 1.0000
    Epoch 6290/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2033 - binary_accuracy: 1.0000
    Epoch 6291/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2033 - binary_accuracy: 1.0000
    Epoch 6292/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2033 - binary_accuracy: 1.0000
    Epoch 6293/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2032 - binary_accuracy: 1.0000
    Epoch 6294/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2032 - binary_accuracy: 1.0000
    Epoch 6295/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2032 - binary_accuracy: 1.0000
    Epoch 6296/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2032 - binary_accuracy: 1.0000
    Epoch 6297/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2031 - binary_accuracy: 1.0000
    Epoch 6298/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2031 - binary_accuracy: 1.0000
    Epoch 6299/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2031 - binary_accuracy: 1.0000
    Epoch 6300/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2031 - binary_accuracy: 1.0000
    Epoch 6301/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2031 - binary_accuracy: 1.0000
    Epoch 6302/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2030 - binary_accuracy: 1.0000
    Epoch 6303/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2030 - binary_accuracy: 1.0000
    Epoch 6304/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2030 - binary_accuracy: 1.0000
    Epoch 6305/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2030 - binary_accuracy: 1.0000
    Epoch 6306/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2030 - binary_accuracy: 1.0000
    Epoch 6307/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2029 - binary_accuracy: 1.0000
    Epoch 6308/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2029 - binary_accuracy: 1.0000
    Epoch 6309/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2029 - binary_accuracy: 1.0000
    Epoch 6310/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2029 - binary_accuracy: 1.0000
    Epoch 6311/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2028 - binary_accuracy: 1.0000
    Epoch 6312/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2028 - binary_accuracy: 1.0000
    Epoch 6313/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2028 - binary_accuracy: 1.0000
    Epoch 6314/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2028 - binary_accuracy: 1.0000
    Epoch 6315/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2028 - binary_accuracy: 1.0000
    Epoch 6316/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2027 - binary_accuracy: 1.0000
    Epoch 6317/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2027 - binary_accuracy: 1.0000
    Epoch 6318/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2027 - binary_accuracy: 1.0000
    Epoch 6319/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2027 - binary_accuracy: 1.0000
    Epoch 6320/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2027 - binary_accuracy: 1.0000
    Epoch 6321/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2026 - binary_accuracy: 1.0000
    Epoch 6322/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2026 - binary_accuracy: 1.0000
    Epoch 6323/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2026 - binary_accuracy: 1.0000
    Epoch 6324/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2026 - binary_accuracy: 1.0000
    Epoch 6325/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2026 - binary_accuracy: 1.0000
    Epoch 6326/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2025 - binary_accuracy: 1.0000
    Epoch 6327/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2025 - binary_accuracy: 1.0000
    Epoch 6328/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2025 - binary_accuracy: 1.0000
    Epoch 6329/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2025 - binary_accuracy: 1.0000
    Epoch 6330/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2024 - binary_accuracy: 1.0000
    Epoch 6331/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2024 - binary_accuracy: 1.0000
    Epoch 6332/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2024 - binary_accuracy: 1.0000
    Epoch 6333/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2024 - binary_accuracy: 1.0000
    Epoch 6334/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2024 - binary_accuracy: 1.0000
    Epoch 6335/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2023 - binary_accuracy: 1.0000
    Epoch 6336/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2023 - binary_accuracy: 1.0000
    Epoch 6337/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2023 - binary_accuracy: 1.0000
    Epoch 6338/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2023 - binary_accuracy: 1.0000
    Epoch 6339/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2023 - binary_accuracy: 1.0000
    Epoch 6340/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2022 - binary_accuracy: 1.0000
    Epoch 6341/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2022 - binary_accuracy: 1.0000
    Epoch 6342/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2022 - binary_accuracy: 1.0000
    Epoch 6343/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2022 - binary_accuracy: 1.0000
    Epoch 6344/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2021 - binary_accuracy: 1.0000
    Epoch 6345/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2021 - binary_accuracy: 1.0000
    Epoch 6346/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2021 - binary_accuracy: 1.0000
    Epoch 6347/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2021 - binary_accuracy: 1.0000
    Epoch 6348/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2021 - binary_accuracy: 1.0000
    Epoch 6349/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2020 - binary_accuracy: 1.0000
    Epoch 6350/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2020 - binary_accuracy: 1.0000
    Epoch 6351/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2020 - binary_accuracy: 1.0000
    Epoch 6352/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2020 - binary_accuracy: 1.0000
    Epoch 6353/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2020 - binary_accuracy: 1.0000
    Epoch 6354/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2019 - binary_accuracy: 1.0000
    Epoch 6355/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2019 - binary_accuracy: 1.0000
    Epoch 6356/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.2019 - binary_accuracy: 1.0000
    Epoch 6357/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2019 - binary_accuracy: 1.0000
    Epoch 6358/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2019 - binary_accuracy: 1.0000
    Epoch 6359/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2018 - binary_accuracy: 1.0000
    Epoch 6360/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2018 - binary_accuracy: 1.0000
    Epoch 6361/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2018 - binary_accuracy: 1.0000
    Epoch 6362/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2018 - binary_accuracy: 1.0000
    Epoch 6363/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2017 - binary_accuracy: 1.0000
    Epoch 6364/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2017 - binary_accuracy: 1.0000
    Epoch 6365/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2017 - binary_accuracy: 1.0000
    Epoch 6366/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2017 - binary_accuracy: 1.0000
    Epoch 6367/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2017 - binary_accuracy: 1.0000
    Epoch 6368/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2016 - binary_accuracy: 1.0000
    Epoch 6369/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2016 - binary_accuracy: 1.0000
    Epoch 6370/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2016 - binary_accuracy: 1.0000
    Epoch 6371/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2016 - binary_accuracy: 1.0000
    Epoch 6372/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2016 - binary_accuracy: 1.0000
    Epoch 6373/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2015 - binary_accuracy: 1.0000
    Epoch 6374/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2015 - binary_accuracy: 1.0000
    Epoch 6375/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2015 - binary_accuracy: 1.0000
    Epoch 6376/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2015 - binary_accuracy: 1.0000
    Epoch 6377/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2014 - binary_accuracy: 1.0000
    Epoch 6378/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2014 - binary_accuracy: 1.0000
    Epoch 6379/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2014 - binary_accuracy: 1.0000
    Epoch 6380/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2014 - binary_accuracy: 1.0000
    Epoch 6381/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2014 - binary_accuracy: 1.0000
    Epoch 6382/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2013 - binary_accuracy: 1.0000
    Epoch 6383/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2013 - binary_accuracy: 1.0000
    Epoch 6384/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2013 - binary_accuracy: 1.0000
    Epoch 6385/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2013 - binary_accuracy: 1.0000
    Epoch 6386/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2013 - binary_accuracy: 1.0000
    Epoch 6387/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2012 - binary_accuracy: 1.0000
    Epoch 6388/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2012 - binary_accuracy: 1.0000
    Epoch 6389/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2012 - binary_accuracy: 1.0000
    Epoch 6390/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2012 - binary_accuracy: 1.0000
    Epoch 6391/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2012 - binary_accuracy: 1.0000
    Epoch 6392/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2011 - binary_accuracy: 1.0000
    Epoch 6393/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2011 - binary_accuracy: 1.0000
    Epoch 6394/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2011 - binary_accuracy: 1.0000
    Epoch 6395/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2011 - binary_accuracy: 1.0000
    Epoch 6396/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2011 - binary_accuracy: 1.0000
    Epoch 6397/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2010 - binary_accuracy: 1.0000
    Epoch 6398/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2010 - binary_accuracy: 1.0000
    Epoch 6399/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2010 - binary_accuracy: 1.0000
    Epoch 6400/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2010 - binary_accuracy: 1.0000
    Epoch 6401/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2009 - binary_accuracy: 1.0000
    Epoch 6402/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2009 - binary_accuracy: 1.0000
    Epoch 6403/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2009 - binary_accuracy: 1.0000
    Epoch 6404/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2009 - binary_accuracy: 1.0000
    Epoch 6405/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2009 - binary_accuracy: 1.0000
    Epoch 6406/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2008 - binary_accuracy: 1.0000
    Epoch 6407/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2008 - binary_accuracy: 1.0000
    Epoch 6408/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2008 - binary_accuracy: 1.0000
    Epoch 6409/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2008 - binary_accuracy: 1.0000
    Epoch 6410/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2008 - binary_accuracy: 1.0000
    Epoch 6411/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2007 - binary_accuracy: 1.0000
    Epoch 6412/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2007 - binary_accuracy: 1.0000
    Epoch 6413/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2007 - binary_accuracy: 1.0000
    Epoch 6414/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2007 - binary_accuracy: 1.0000
    Epoch 6415/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2007 - binary_accuracy: 1.0000
    Epoch 6416/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2006 - binary_accuracy: 1.0000
    Epoch 6417/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2006 - binary_accuracy: 1.0000
    Epoch 6418/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2006 - binary_accuracy: 1.0000
    Epoch 6419/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2006 - binary_accuracy: 1.0000
    Epoch 6420/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2005 - binary_accuracy: 1.0000
    Epoch 6421/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2005 - binary_accuracy: 1.0000
    Epoch 6422/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2005 - binary_accuracy: 1.0000
    Epoch 6423/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.2005 - binary_accuracy: 1.0000
    Epoch 6424/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.2005 - binary_accuracy: 1.0000
    Epoch 6425/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2004 - binary_accuracy: 1.0000
    Epoch 6426/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2004 - binary_accuracy: 1.0000
    Epoch 6427/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2004 - binary_accuracy: 1.0000
    Epoch 6428/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2004 - binary_accuracy: 1.0000
    Epoch 6429/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2004 - binary_accuracy: 1.0000
    Epoch 6430/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2003 - binary_accuracy: 1.0000
    Epoch 6431/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2003 - binary_accuracy: 1.0000
    Epoch 6432/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.2003 - binary_accuracy: 1.0000
    Epoch 6433/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.2003 - binary_accuracy: 1.0000
    Epoch 6434/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2003 - binary_accuracy: 1.0000
    Epoch 6435/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2002 - binary_accuracy: 1.0000
    Epoch 6436/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2002 - binary_accuracy: 1.0000
    Epoch 6437/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2002 - binary_accuracy: 1.0000
    Epoch 6438/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2002 - binary_accuracy: 1.0000
    Epoch 6439/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2002 - binary_accuracy: 1.0000
    Epoch 6440/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2001 - binary_accuracy: 1.0000
    Epoch 6441/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2001 - binary_accuracy: 1.0000
    Epoch 6442/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2001 - binary_accuracy: 1.0000
    Epoch 6443/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2001 - binary_accuracy: 1.0000
    Epoch 6444/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.2000 - binary_accuracy: 1.0000
    Epoch 6445/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2000 - binary_accuracy: 1.0000
    Epoch 6446/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.2000 - binary_accuracy: 1.0000
    Epoch 6447/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.2000 - binary_accuracy: 1.0000
    Epoch 6448/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.2000 - binary_accuracy: 1.0000
    Epoch 6449/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1999 - binary_accuracy: 1.0000
    Epoch 6450/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1999 - binary_accuracy: 1.0000
    Epoch 6451/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1999 - binary_accuracy: 1.0000
    Epoch 6452/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1999 - binary_accuracy: 1.0000
    Epoch 6453/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1999 - binary_accuracy: 1.0000
    Epoch 6454/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1998 - binary_accuracy: 1.0000
    Epoch 6455/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1998 - binary_accuracy: 1.0000
    Epoch 6456/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1998 - binary_accuracy: 1.0000
    Epoch 6457/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1998 - binary_accuracy: 1.0000
    Epoch 6458/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1998 - binary_accuracy: 1.0000
    Epoch 6459/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1997 - binary_accuracy: 1.0000
    Epoch 6460/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1997 - binary_accuracy: 1.0000
    Epoch 6461/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1997 - binary_accuracy: 1.0000
    Epoch 6462/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1997 - binary_accuracy: 1.0000
    Epoch 6463/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1997 - binary_accuracy: 1.0000
    Epoch 6464/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1996 - binary_accuracy: 1.0000
    Epoch 6465/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1996 - binary_accuracy: 1.0000
    Epoch 6466/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1996 - binary_accuracy: 1.0000
    Epoch 6467/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1996 - binary_accuracy: 1.0000
    Epoch 6468/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1996 - binary_accuracy: 1.0000
    Epoch 6469/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1995 - binary_accuracy: 1.0000
    Epoch 6470/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1995 - binary_accuracy: 1.0000
    Epoch 6471/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1995 - binary_accuracy: 1.0000
    Epoch 6472/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1995 - binary_accuracy: 1.0000
    Epoch 6473/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1994 - binary_accuracy: 1.0000
    Epoch 6474/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1994 - binary_accuracy: 1.0000
    Epoch 6475/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1994 - binary_accuracy: 1.0000
    Epoch 6476/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1994 - binary_accuracy: 1.0000
    Epoch 6477/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1994 - binary_accuracy: 1.0000
    Epoch 6478/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1993 - binary_accuracy: 1.0000
    Epoch 6479/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1993 - binary_accuracy: 1.0000
    Epoch 6480/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1993 - binary_accuracy: 1.0000
    Epoch 6481/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1993 - binary_accuracy: 1.0000
    Epoch 6482/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1993 - binary_accuracy: 1.0000
    Epoch 6483/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1992 - binary_accuracy: 1.0000
    Epoch 6484/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1992 - binary_accuracy: 1.0000
    Epoch 6485/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1992 - binary_accuracy: 1.0000
    Epoch 6486/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1992 - binary_accuracy: 1.0000
    Epoch 6487/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1992 - binary_accuracy: 1.0000
    Epoch 6488/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1991 - binary_accuracy: 1.0000
    Epoch 6489/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1991 - binary_accuracy: 1.0000
    Epoch 6490/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1991 - binary_accuracy: 1.0000
    Epoch 6491/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1991 - binary_accuracy: 1.0000
    Epoch 6492/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1991 - binary_accuracy: 1.0000
    Epoch 6493/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1990 - binary_accuracy: 1.0000
    Epoch 6494/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1990 - binary_accuracy: 1.0000
    Epoch 6495/7000
    1/1 [==============================] - 0s 12ms/step - loss: 0.1990 - binary_accuracy: 1.0000
    Epoch 6496/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1990 - binary_accuracy: 1.0000
    Epoch 6497/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1990 - binary_accuracy: 1.0000
    Epoch 6498/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1989 - binary_accuracy: 1.0000
    Epoch 6499/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1989 - binary_accuracy: 1.0000
    Epoch 6500/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1989 - binary_accuracy: 1.0000
    Epoch 6501/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1989 - binary_accuracy: 1.0000
    Epoch 6502/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1989 - binary_accuracy: 1.0000
    Epoch 6503/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1988 - binary_accuracy: 1.0000
    Epoch 6504/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.1988 - binary_accuracy: 1.0000
    Epoch 6505/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1988 - binary_accuracy: 1.0000
    Epoch 6506/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1988 - binary_accuracy: 1.0000
    Epoch 6507/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1988 - binary_accuracy: 1.0000
    Epoch 6508/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1987 - binary_accuracy: 1.0000
    Epoch 6509/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1987 - binary_accuracy: 1.0000
    Epoch 6510/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1987 - binary_accuracy: 1.0000
    Epoch 6511/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1987 - binary_accuracy: 1.0000
    Epoch 6512/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1986 - binary_accuracy: 1.0000
    Epoch 6513/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1986 - binary_accuracy: 1.0000
    Epoch 6514/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1986 - binary_accuracy: 1.0000
    Epoch 6515/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1986 - binary_accuracy: 1.0000
    Epoch 6516/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1986 - binary_accuracy: 1.0000
    Epoch 6517/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1985 - binary_accuracy: 1.0000
    Epoch 6518/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1985 - binary_accuracy: 1.0000
    Epoch 6519/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1985 - binary_accuracy: 1.0000
    Epoch 6520/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1985 - binary_accuracy: 1.0000
    Epoch 6521/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1985 - binary_accuracy: 1.0000
    Epoch 6522/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1984 - binary_accuracy: 1.0000
    Epoch 6523/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1984 - binary_accuracy: 1.0000
    Epoch 6524/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1984 - binary_accuracy: 1.0000
    Epoch 6525/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1984 - binary_accuracy: 1.0000
    Epoch 6526/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1984 - binary_accuracy: 1.0000
    Epoch 6527/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1983 - binary_accuracy: 1.0000
    Epoch 6528/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1983 - binary_accuracy: 1.0000
    Epoch 6529/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1983 - binary_accuracy: 1.0000
    Epoch 6530/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1983 - binary_accuracy: 1.0000
    Epoch 6531/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1983 - binary_accuracy: 1.0000
    Epoch 6532/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1982 - binary_accuracy: 1.0000
    Epoch 6533/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1982 - binary_accuracy: 1.0000
    Epoch 6534/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1982 - binary_accuracy: 1.0000
    Epoch 6535/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1982 - binary_accuracy: 1.0000
    Epoch 6536/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1982 - binary_accuracy: 1.0000
    Epoch 6537/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1981 - binary_accuracy: 1.0000
    Epoch 6538/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1981 - binary_accuracy: 1.0000
    Epoch 6539/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1981 - binary_accuracy: 1.0000
    Epoch 6540/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1981 - binary_accuracy: 1.0000
    Epoch 6541/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1981 - binary_accuracy: 1.0000
    Epoch 6542/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1980 - binary_accuracy: 1.0000
    Epoch 6543/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1980 - binary_accuracy: 1.0000
    Epoch 6544/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1980 - binary_accuracy: 1.0000
    Epoch 6545/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1980 - binary_accuracy: 1.0000
    Epoch 6546/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1980 - binary_accuracy: 1.0000
    Epoch 6547/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1979 - binary_accuracy: 1.0000
    Epoch 6548/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1979 - binary_accuracy: 1.0000
    Epoch 6549/7000
    1/1 [==============================] - 0s 16ms/step - loss: 0.1979 - binary_accuracy: 1.0000
    Epoch 6550/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1979 - binary_accuracy: 1.0000
    Epoch 6551/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1979 - binary_accuracy: 1.0000
    Epoch 6552/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1978 - binary_accuracy: 1.0000
    Epoch 6553/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1978 - binary_accuracy: 1.0000
    Epoch 6554/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1978 - binary_accuracy: 1.0000
    Epoch 6555/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1978 - binary_accuracy: 1.0000
    Epoch 6556/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1978 - binary_accuracy: 1.0000
    Epoch 6557/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1977 - binary_accuracy: 1.0000
    Epoch 6558/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.1977 - binary_accuracy: 1.0000
    Epoch 6559/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1977 - binary_accuracy: 1.0000
    Epoch 6560/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1977 - binary_accuracy: 1.0000
    Epoch 6561/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.1977 - binary_accuracy: 1.0000
    Epoch 6562/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1976 - binary_accuracy: 1.0000
    Epoch 6563/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1976 - binary_accuracy: 1.0000
    Epoch 6564/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1976 - binary_accuracy: 1.0000
    Epoch 6565/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1976 - binary_accuracy: 1.0000
    Epoch 6566/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1975 - binary_accuracy: 1.0000
    Epoch 6567/7000
    1/1 [==============================] - 0s 9ms/step - loss: 0.1975 - binary_accuracy: 1.0000
    Epoch 6568/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1975 - binary_accuracy: 1.0000
    Epoch 6569/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1975 - binary_accuracy: 1.0000
    Epoch 6570/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1975 - binary_accuracy: 1.0000
    Epoch 6571/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1974 - binary_accuracy: 1.0000
    Epoch 6572/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1974 - binary_accuracy: 1.0000
    Epoch 6573/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1974 - binary_accuracy: 1.0000
    Epoch 6574/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1974 - binary_accuracy: 1.0000
    Epoch 6575/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1974 - binary_accuracy: 1.0000
    Epoch 6576/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1973 - binary_accuracy: 1.0000
    Epoch 6577/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1973 - binary_accuracy: 1.0000
    Epoch 6578/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1973 - binary_accuracy: 1.0000
    Epoch 6579/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1973 - binary_accuracy: 1.0000
    Epoch 6580/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1973 - binary_accuracy: 1.0000
    Epoch 6581/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1972 - binary_accuracy: 1.0000
    Epoch 6582/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1972 - binary_accuracy: 1.0000
    Epoch 6583/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1972 - binary_accuracy: 1.0000
    Epoch 6584/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1972 - binary_accuracy: 1.0000
    Epoch 6585/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1972 - binary_accuracy: 1.0000
    Epoch 6586/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1971 - binary_accuracy: 1.0000
    Epoch 6587/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.1971 - binary_accuracy: 1.0000
    Epoch 6588/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1971 - binary_accuracy: 1.0000
    Epoch 6589/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1971 - binary_accuracy: 1.0000
    Epoch 6590/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1971 - binary_accuracy: 1.0000
    Epoch 6591/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1970 - binary_accuracy: 1.0000
    Epoch 6592/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1970 - binary_accuracy: 1.0000
    Epoch 6593/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1970 - binary_accuracy: 1.0000
    Epoch 6594/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1970 - binary_accuracy: 1.0000
    Epoch 6595/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1970 - binary_accuracy: 1.0000
    Epoch 6596/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1969 - binary_accuracy: 1.0000
    Epoch 6597/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1969 - binary_accuracy: 1.0000
    Epoch 6598/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1969 - binary_accuracy: 1.0000
    Epoch 6599/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1969 - binary_accuracy: 1.0000
    Epoch 6600/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1969 - binary_accuracy: 1.0000
    Epoch 6601/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1968 - binary_accuracy: 1.0000
    Epoch 6602/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1968 - binary_accuracy: 1.0000
    Epoch 6603/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1968 - binary_accuracy: 1.0000
    Epoch 6604/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1968 - binary_accuracy: 1.0000
    Epoch 6605/7000
    1/1 [==============================] - 0s 14ms/step - loss: 0.1968 - binary_accuracy: 1.0000
    Epoch 6606/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1967 - binary_accuracy: 1.0000
    Epoch 6607/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1967 - binary_accuracy: 1.0000
    Epoch 6608/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1967 - binary_accuracy: 1.0000
    Epoch 6609/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1967 - binary_accuracy: 1.0000
    Epoch 6610/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1967 - binary_accuracy: 1.0000
    Epoch 6611/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1966 - binary_accuracy: 1.0000
    Epoch 6612/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1966 - binary_accuracy: 1.0000
    Epoch 6613/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1966 - binary_accuracy: 1.0000
    Epoch 6614/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1966 - binary_accuracy: 1.0000
    Epoch 6615/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1966 - binary_accuracy: 1.0000
    Epoch 6616/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1965 - binary_accuracy: 1.0000
    Epoch 6617/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1965 - binary_accuracy: 1.0000
    Epoch 6618/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1965 - binary_accuracy: 1.0000
    Epoch 6619/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1965 - binary_accuracy: 1.0000
    Epoch 6620/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1965 - binary_accuracy: 1.0000
    Epoch 6621/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1964 - binary_accuracy: 1.0000
    Epoch 6622/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1964 - binary_accuracy: 1.0000
    Epoch 6623/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.1964 - binary_accuracy: 1.0000
    Epoch 6624/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1964 - binary_accuracy: 1.0000
    Epoch 6625/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1964 - binary_accuracy: 1.0000
    Epoch 6626/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1963 - binary_accuracy: 1.0000
    Epoch 6627/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1963 - binary_accuracy: 1.0000
    Epoch 6628/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1963 - binary_accuracy: 1.0000
    Epoch 6629/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1963 - binary_accuracy: 1.0000
    Epoch 6630/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1963 - binary_accuracy: 1.0000
    Epoch 6631/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1962 - binary_accuracy: 1.0000
    Epoch 6632/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1962 - binary_accuracy: 1.0000
    Epoch 6633/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1962 - binary_accuracy: 1.0000
    Epoch 6634/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1962 - binary_accuracy: 1.0000
    Epoch 6635/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1962 - binary_accuracy: 1.0000
    Epoch 6636/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1961 - binary_accuracy: 1.0000
    Epoch 6637/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1961 - binary_accuracy: 1.0000
    Epoch 6638/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1961 - binary_accuracy: 1.0000
    Epoch 6639/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1961 - binary_accuracy: 1.0000
    Epoch 6640/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1961 - binary_accuracy: 1.0000
    Epoch 6641/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1960 - binary_accuracy: 1.0000
    Epoch 6642/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1960 - binary_accuracy: 1.0000
    Epoch 6643/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1960 - binary_accuracy: 1.0000
    Epoch 6644/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1960 - binary_accuracy: 1.0000
    Epoch 6645/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1960 - binary_accuracy: 1.0000
    Epoch 6646/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1959 - binary_accuracy: 1.0000
    Epoch 6647/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1959 - binary_accuracy: 1.0000
    Epoch 6648/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1959 - binary_accuracy: 1.0000
    Epoch 6649/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1959 - binary_accuracy: 1.0000
    Epoch 6650/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1959 - binary_accuracy: 1.0000
    Epoch 6651/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.1958 - binary_accuracy: 1.0000
    Epoch 6652/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1958 - binary_accuracy: 1.0000
    Epoch 6653/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1958 - binary_accuracy: 1.0000
    Epoch 6654/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1958 - binary_accuracy: 1.0000
    Epoch 6655/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1958 - binary_accuracy: 1.0000
    Epoch 6656/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1957 - binary_accuracy: 1.0000
    Epoch 6657/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1957 - binary_accuracy: 1.0000
    Epoch 6658/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1957 - binary_accuracy: 1.0000
    Epoch 6659/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1957 - binary_accuracy: 1.0000
    Epoch 6660/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1957 - binary_accuracy: 1.0000
    Epoch 6661/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.1956 - binary_accuracy: 1.0000
    Epoch 6662/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1956 - binary_accuracy: 1.0000
    Epoch 6663/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.1956 - binary_accuracy: 1.0000
    Epoch 6664/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1956 - binary_accuracy: 1.0000
    Epoch 6665/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1956 - binary_accuracy: 1.0000
    Epoch 6666/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1955 - binary_accuracy: 1.0000
    Epoch 6667/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1955 - binary_accuracy: 1.0000
    Epoch 6668/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1955 - binary_accuracy: 1.0000
    Epoch 6669/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1955 - binary_accuracy: 1.0000
    Epoch 6670/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1955 - binary_accuracy: 1.0000
    Epoch 6671/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1954 - binary_accuracy: 1.0000
    Epoch 6672/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1954 - binary_accuracy: 1.0000
    Epoch 6673/7000
    1/1 [==============================] - 0s 15ms/step - loss: 0.1954 - binary_accuracy: 1.0000
    Epoch 6674/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.1954 - binary_accuracy: 1.0000
    Epoch 6675/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1954 - binary_accuracy: 1.0000
    Epoch 6676/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1954 - binary_accuracy: 1.0000
    Epoch 6677/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1953 - binary_accuracy: 1.0000
    Epoch 6678/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1953 - binary_accuracy: 1.0000
    Epoch 6679/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1953 - binary_accuracy: 1.0000
    Epoch 6680/7000
    1/1 [==============================] - 0s 24ms/step - loss: 0.1953 - binary_accuracy: 1.0000
    Epoch 6681/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1953 - binary_accuracy: 1.0000
    Epoch 6682/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1952 - binary_accuracy: 1.0000
    Epoch 6683/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1952 - binary_accuracy: 1.0000
    Epoch 6684/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1952 - binary_accuracy: 1.0000
    Epoch 6685/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1952 - binary_accuracy: 1.0000
    Epoch 6686/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1952 - binary_accuracy: 1.0000
    Epoch 6687/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1951 - binary_accuracy: 1.0000
    Epoch 6688/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1951 - binary_accuracy: 1.0000
    Epoch 6689/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1951 - binary_accuracy: 1.0000
    Epoch 6690/7000
    1/1 [==============================] - 0s 11ms/step - loss: 0.1951 - binary_accuracy: 1.0000
    Epoch 6691/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1951 - binary_accuracy: 1.0000
    Epoch 6692/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1950 - binary_accuracy: 1.0000
    Epoch 6693/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1950 - binary_accuracy: 1.0000
    Epoch 6694/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1950 - binary_accuracy: 1.0000
    Epoch 6695/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1950 - binary_accuracy: 1.0000
    Epoch 6696/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1950 - binary_accuracy: 1.0000
    Epoch 6697/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1949 - binary_accuracy: 1.0000
    Epoch 6698/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1949 - binary_accuracy: 1.0000
    Epoch 6699/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1949 - binary_accuracy: 1.0000
    Epoch 6700/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.1949 - binary_accuracy: 1.0000
    Epoch 6701/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.1949 - binary_accuracy: 1.0000
    Epoch 6702/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1948 - binary_accuracy: 1.0000
    Epoch 6703/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1948 - binary_accuracy: 1.0000
    Epoch 6704/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1948 - binary_accuracy: 1.0000
    Epoch 6705/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1948 - binary_accuracy: 1.0000
    Epoch 6706/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1948 - binary_accuracy: 1.0000
    Epoch 6707/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1947 - binary_accuracy: 1.0000
    Epoch 6708/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1947 - binary_accuracy: 1.0000
    Epoch 6709/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1947 - binary_accuracy: 1.0000
    Epoch 6710/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1947 - binary_accuracy: 1.0000
    Epoch 6711/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1947 - binary_accuracy: 1.0000
    Epoch 6712/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1946 - binary_accuracy: 1.0000
    Epoch 6713/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1946 - binary_accuracy: 1.0000
    Epoch 6714/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1946 - binary_accuracy: 1.0000
    Epoch 6715/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1946 - binary_accuracy: 1.0000
    Epoch 6716/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1946 - binary_accuracy: 1.0000
    Epoch 6717/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1945 - binary_accuracy: 1.0000
    Epoch 6718/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1945 - binary_accuracy: 1.0000
    Epoch 6719/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1945 - binary_accuracy: 1.0000
    Epoch 6720/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1945 - binary_accuracy: 1.0000
    Epoch 6721/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1945 - binary_accuracy: 1.0000
    Epoch 6722/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1944 - binary_accuracy: 1.0000
    Epoch 6723/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1944 - binary_accuracy: 1.0000
    Epoch 6724/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1944 - binary_accuracy: 1.0000
    Epoch 6725/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1944 - binary_accuracy: 1.0000
    Epoch 6726/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1944 - binary_accuracy: 1.0000
    Epoch 6727/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1943 - binary_accuracy: 1.0000
    Epoch 6728/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1943 - binary_accuracy: 1.0000
    Epoch 6729/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1943 - binary_accuracy: 1.0000
    Epoch 6730/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1943 - binary_accuracy: 1.0000
    Epoch 6731/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.1943 - binary_accuracy: 1.0000
    Epoch 6732/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1942 - binary_accuracy: 1.0000
    Epoch 6733/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1942 - binary_accuracy: 1.0000
    Epoch 6734/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1942 - binary_accuracy: 1.0000
    Epoch 6735/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1942 - binary_accuracy: 1.0000
    Epoch 6736/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1942 - binary_accuracy: 1.0000
    Epoch 6737/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1942 - binary_accuracy: 1.0000
    Epoch 6738/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1941 - binary_accuracy: 1.0000
    Epoch 6739/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1941 - binary_accuracy: 1.0000
    Epoch 6740/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1941 - binary_accuracy: 1.0000
    Epoch 6741/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1941 - binary_accuracy: 1.0000
    Epoch 6742/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1941 - binary_accuracy: 1.0000
    Epoch 6743/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1940 - binary_accuracy: 1.0000
    Epoch 6744/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1940 - binary_accuracy: 1.0000
    Epoch 6745/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1940 - binary_accuracy: 1.0000
    Epoch 6746/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1940 - binary_accuracy: 1.0000
    Epoch 6747/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1940 - binary_accuracy: 1.0000
    Epoch 6748/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1939 - binary_accuracy: 1.0000
    Epoch 6749/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1939 - binary_accuracy: 1.0000
    Epoch 6750/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1939 - binary_accuracy: 1.0000
    Epoch 6751/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1939 - binary_accuracy: 1.0000
    Epoch 6752/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1939 - binary_accuracy: 1.0000
    Epoch 6753/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1938 - binary_accuracy: 1.0000
    Epoch 6754/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1938 - binary_accuracy: 1.0000
    Epoch 6755/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1938 - binary_accuracy: 1.0000
    Epoch 6756/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1938 - binary_accuracy: 1.0000
    Epoch 6757/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1938 - binary_accuracy: 1.0000
    Epoch 6758/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1937 - binary_accuracy: 1.0000
    Epoch 6759/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1937 - binary_accuracy: 1.0000
    Epoch 6760/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1937 - binary_accuracy: 1.0000
    Epoch 6761/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1937 - binary_accuracy: 1.0000
    Epoch 6762/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1937 - binary_accuracy: 1.0000
    Epoch 6763/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1936 - binary_accuracy: 1.0000
    Epoch 6764/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1936 - binary_accuracy: 1.0000
    Epoch 6765/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1936 - binary_accuracy: 1.0000
    Epoch 6766/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1936 - binary_accuracy: 1.0000
    Epoch 6767/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1936 - binary_accuracy: 1.0000
    Epoch 6768/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1935 - binary_accuracy: 1.0000
    Epoch 6769/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1935 - binary_accuracy: 1.0000
    Epoch 6770/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1935 - binary_accuracy: 1.0000
    Epoch 6771/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1935 - binary_accuracy: 1.0000
    Epoch 6772/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1935 - binary_accuracy: 1.0000
    Epoch 6773/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1935 - binary_accuracy: 1.0000
    Epoch 6774/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1934 - binary_accuracy: 1.0000
    Epoch 6775/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1934 - binary_accuracy: 1.0000
    Epoch 6776/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1934 - binary_accuracy: 1.0000
    Epoch 6777/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1934 - binary_accuracy: 1.0000
    Epoch 6778/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1934 - binary_accuracy: 1.0000
    Epoch 6779/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1933 - binary_accuracy: 1.0000
    Epoch 6780/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1933 - binary_accuracy: 1.0000
    Epoch 6781/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1933 - binary_accuracy: 1.0000
    Epoch 6782/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1933 - binary_accuracy: 1.0000
    Epoch 6783/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1933 - binary_accuracy: 1.0000
    Epoch 6784/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1932 - binary_accuracy: 1.0000
    Epoch 6785/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1932 - binary_accuracy: 1.0000
    Epoch 6786/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1932 - binary_accuracy: 1.0000
    Epoch 6787/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1932 - binary_accuracy: 1.0000
    Epoch 6788/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1932 - binary_accuracy: 1.0000
    Epoch 6789/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1931 - binary_accuracy: 1.0000
    Epoch 6790/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1931 - binary_accuracy: 1.0000
    Epoch 6791/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1931 - binary_accuracy: 1.0000
    Epoch 6792/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1931 - binary_accuracy: 1.0000
    Epoch 6793/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1931 - binary_accuracy: 1.0000
    Epoch 6794/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1930 - binary_accuracy: 1.0000
    Epoch 6795/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1930 - binary_accuracy: 1.0000
    Epoch 6796/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1930 - binary_accuracy: 1.0000
    Epoch 6797/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1930 - binary_accuracy: 1.0000
    Epoch 6798/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1930 - binary_accuracy: 1.0000
    Epoch 6799/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1929 - binary_accuracy: 1.0000
    Epoch 6800/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1929 - binary_accuracy: 1.0000
    Epoch 6801/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1929 - binary_accuracy: 1.0000
    Epoch 6802/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1929 - binary_accuracy: 1.0000
    Epoch 6803/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1929 - binary_accuracy: 1.0000
    Epoch 6804/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1929 - binary_accuracy: 1.0000
    Epoch 6805/7000
    1/1 [==============================] - 0s 13ms/step - loss: 0.1928 - binary_accuracy: 1.0000
    Epoch 6806/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1928 - binary_accuracy: 1.0000
    Epoch 6807/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1928 - binary_accuracy: 1.0000
    Epoch 6808/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1928 - binary_accuracy: 1.0000
    Epoch 6809/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1928 - binary_accuracy: 1.0000
    Epoch 6810/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1927 - binary_accuracy: 1.0000
    Epoch 6811/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1927 - binary_accuracy: 1.0000
    Epoch 6812/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1927 - binary_accuracy: 1.0000
    Epoch 6813/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1927 - binary_accuracy: 1.0000
    Epoch 6814/7000
    1/1 [==============================] - 0s 8ms/step - loss: 0.1927 - binary_accuracy: 1.0000
    Epoch 6815/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1926 - binary_accuracy: 1.0000
    Epoch 6816/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1926 - binary_accuracy: 1.0000
    Epoch 6817/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1926 - binary_accuracy: 1.0000
    Epoch 6818/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1926 - binary_accuracy: 1.0000
    Epoch 6819/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1926 - binary_accuracy: 1.0000
    Epoch 6820/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1925 - binary_accuracy: 1.0000
    Epoch 6821/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1925 - binary_accuracy: 1.0000
    Epoch 6822/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1925 - binary_accuracy: 1.0000
    Epoch 6823/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1925 - binary_accuracy: 1.0000
    Epoch 6824/7000
    1/1 [==============================] - 0s 10ms/step - loss: 0.1925 - binary_accuracy: 1.0000
    Epoch 6825/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1924 - binary_accuracy: 1.0000
    Epoch 6826/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1924 - binary_accuracy: 1.0000
    Epoch 6827/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1924 - binary_accuracy: 1.0000
    Epoch 6828/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1924 - binary_accuracy: 1.0000
    Epoch 6829/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1924 - binary_accuracy: 1.0000
    Epoch 6830/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1924 - binary_accuracy: 1.0000
    Epoch 6831/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1923 - binary_accuracy: 1.0000
    Epoch 6832/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1923 - binary_accuracy: 1.0000
    Epoch 6833/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1923 - binary_accuracy: 1.0000
    Epoch 6834/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1923 - binary_accuracy: 1.0000
    Epoch 6835/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1923 - binary_accuracy: 1.0000
    Epoch 6836/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1922 - binary_accuracy: 1.0000
    Epoch 6837/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1922 - binary_accuracy: 1.0000
    Epoch 6838/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1922 - binary_accuracy: 1.0000
    Epoch 6839/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1922 - binary_accuracy: 1.0000
    Epoch 6840/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1922 - binary_accuracy: 1.0000
    Epoch 6841/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1921 - binary_accuracy: 1.0000
    Epoch 6842/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1921 - binary_accuracy: 1.0000
    Epoch 6843/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1921 - binary_accuracy: 1.0000
    Epoch 6844/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1921 - binary_accuracy: 1.0000
    Epoch 6845/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1921 - binary_accuracy: 1.0000
    Epoch 6846/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1920 - binary_accuracy: 1.0000
    Epoch 6847/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1920 - binary_accuracy: 1.0000
    Epoch 6848/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1920 - binary_accuracy: 1.0000
    Epoch 6849/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1920 - binary_accuracy: 1.0000
    Epoch 6850/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1920 - binary_accuracy: 1.0000
    Epoch 6851/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1920 - binary_accuracy: 1.0000
    Epoch 6852/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1919 - binary_accuracy: 1.0000
    Epoch 6853/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1919 - binary_accuracy: 1.0000
    Epoch 6854/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1919 - binary_accuracy: 1.0000
    Epoch 6855/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1919 - binary_accuracy: 1.0000
    Epoch 6856/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1919 - binary_accuracy: 1.0000
    Epoch 6857/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1918 - binary_accuracy: 1.0000
    Epoch 6858/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1918 - binary_accuracy: 1.0000
    Epoch 6859/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1918 - binary_accuracy: 1.0000
    Epoch 6860/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1918 - binary_accuracy: 1.0000
    Epoch 6861/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1918 - binary_accuracy: 1.0000
    Epoch 6862/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1917 - binary_accuracy: 1.0000
    Epoch 6863/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1917 - binary_accuracy: 1.0000
    Epoch 6864/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1917 - binary_accuracy: 1.0000
    Epoch 6865/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1917 - binary_accuracy: 1.0000
    Epoch 6866/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1917 - binary_accuracy: 1.0000
    Epoch 6867/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1916 - binary_accuracy: 1.0000
    Epoch 6868/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1916 - binary_accuracy: 1.0000
    Epoch 6869/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1916 - binary_accuracy: 1.0000
    Epoch 6870/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1916 - binary_accuracy: 1.0000
    Epoch 6871/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1916 - binary_accuracy: 1.0000
    Epoch 6872/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1916 - binary_accuracy: 1.0000
    Epoch 6873/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1915 - binary_accuracy: 1.0000
    Epoch 6874/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1915 - binary_accuracy: 1.0000
    Epoch 6875/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1915 - binary_accuracy: 1.0000
    Epoch 6876/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1915 - binary_accuracy: 1.0000
    Epoch 6877/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1915 - binary_accuracy: 1.0000
    Epoch 6878/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1914 - binary_accuracy: 1.0000
    Epoch 6879/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1914 - binary_accuracy: 1.0000
    Epoch 6880/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1914 - binary_accuracy: 1.0000
    Epoch 6881/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1914 - binary_accuracy: 1.0000
    Epoch 6882/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1914 - binary_accuracy: 1.0000
    Epoch 6883/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1913 - binary_accuracy: 1.0000
    Epoch 6884/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1913 - binary_accuracy: 1.0000
    Epoch 6885/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1913 - binary_accuracy: 1.0000
    Epoch 6886/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1913 - binary_accuracy: 1.0000
    Epoch 6887/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1913 - binary_accuracy: 1.0000
    Epoch 6888/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1912 - binary_accuracy: 1.0000
    Epoch 6889/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1912 - binary_accuracy: 1.0000
    Epoch 6890/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1912 - binary_accuracy: 1.0000
    Epoch 6891/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1912 - binary_accuracy: 1.0000
    Epoch 6892/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1912 - binary_accuracy: 1.0000
    Epoch 6893/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1912 - binary_accuracy: 1.0000
    Epoch 6894/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1911 - binary_accuracy: 1.0000
    Epoch 6895/7000
    1/1 [==============================] - 0s 7ms/step - loss: 0.1911 - binary_accuracy: 1.0000
    Epoch 6896/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1911 - binary_accuracy: 1.0000
    Epoch 6897/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1911 - binary_accuracy: 1.0000
    Epoch 6898/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1911 - binary_accuracy: 1.0000
    Epoch 6899/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1910 - binary_accuracy: 1.0000
    Epoch 6900/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1910 - binary_accuracy: 1.0000
    Epoch 6901/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1910 - binary_accuracy: 1.0000
    Epoch 6902/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1910 - binary_accuracy: 1.0000
    Epoch 6903/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1910 - binary_accuracy: 1.0000
    Epoch 6904/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1909 - binary_accuracy: 1.0000
    Epoch 6905/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1909 - binary_accuracy: 1.0000
    Epoch 6906/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1909 - binary_accuracy: 1.0000
    Epoch 6907/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1909 - binary_accuracy: 1.0000
    Epoch 6908/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1909 - binary_accuracy: 1.0000
    Epoch 6909/7000
    1/1 [==============================] - 0s 21ms/step - loss: 0.1909 - binary_accuracy: 1.0000
    Epoch 6910/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1908 - binary_accuracy: 1.0000
    Epoch 6911/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1908 - binary_accuracy: 1.0000
    Epoch 6912/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1908 - binary_accuracy: 1.0000
    Epoch 6913/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1908 - binary_accuracy: 1.0000
    Epoch 6914/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1908 - binary_accuracy: 1.0000
    Epoch 6915/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1907 - binary_accuracy: 1.0000
    Epoch 6916/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1907 - binary_accuracy: 1.0000
    Epoch 6917/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1907 - binary_accuracy: 1.0000
    Epoch 6918/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1907 - binary_accuracy: 1.0000
    Epoch 6919/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1907 - binary_accuracy: 1.0000
    Epoch 6920/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1906 - binary_accuracy: 1.0000
    Epoch 6921/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1906 - binary_accuracy: 1.0000
    Epoch 6922/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1906 - binary_accuracy: 1.0000
    Epoch 6923/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1906 - binary_accuracy: 1.0000
    Epoch 6924/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1906 - binary_accuracy: 1.0000
    Epoch 6925/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1906 - binary_accuracy: 1.0000
    Epoch 6926/7000
    1/1 [==============================] - 0s 6ms/step - loss: 0.1905 - binary_accuracy: 1.0000
    Epoch 6927/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1905 - binary_accuracy: 1.0000
    Epoch 6928/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1905 - binary_accuracy: 1.0000
    Epoch 6929/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1905 - binary_accuracy: 1.0000
    Epoch 6930/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1905 - binary_accuracy: 1.0000
    Epoch 6931/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1904 - binary_accuracy: 1.0000
    Epoch 6932/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1904 - binary_accuracy: 1.0000
    Epoch 6933/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1904 - binary_accuracy: 1.0000
    Epoch 6934/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1904 - binary_accuracy: 1.0000
    Epoch 6935/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1904 - binary_accuracy: 1.0000
    Epoch 6936/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1903 - binary_accuracy: 1.0000
    Epoch 6937/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1903 - binary_accuracy: 1.0000
    Epoch 6938/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1903 - binary_accuracy: 1.0000
    Epoch 6939/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1903 - binary_accuracy: 1.0000
    Epoch 6940/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1903 - binary_accuracy: 1.0000
    Epoch 6941/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1903 - binary_accuracy: 1.0000
    Epoch 6942/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1902 - binary_accuracy: 1.0000
    Epoch 6943/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1902 - binary_accuracy: 1.0000
    Epoch 6944/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1902 - binary_accuracy: 1.0000
    Epoch 6945/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1902 - binary_accuracy: 1.0000
    Epoch 6946/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1902 - binary_accuracy: 1.0000
    Epoch 6947/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1901 - binary_accuracy: 1.0000
    Epoch 6948/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1901 - binary_accuracy: 1.0000
    Epoch 6949/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1901 - binary_accuracy: 1.0000
    Epoch 6950/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1901 - binary_accuracy: 1.0000
    Epoch 6951/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1901 - binary_accuracy: 1.0000
    Epoch 6952/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1900 - binary_accuracy: 1.0000
    Epoch 6953/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1900 - binary_accuracy: 1.0000
    Epoch 6954/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1900 - binary_accuracy: 1.0000
    Epoch 6955/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1900 - binary_accuracy: 1.0000
    Epoch 6956/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1900 - binary_accuracy: 1.0000
    Epoch 6957/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1900 - binary_accuracy: 1.0000
    Epoch 6958/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1899 - binary_accuracy: 1.0000
    Epoch 6959/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1899 - binary_accuracy: 1.0000
    Epoch 6960/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1899 - binary_accuracy: 1.0000
    Epoch 6961/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1899 - binary_accuracy: 1.0000
    Epoch 6962/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1899 - binary_accuracy: 1.0000
    Epoch 6963/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1898 - binary_accuracy: 1.0000
    Epoch 6964/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1898 - binary_accuracy: 1.0000
    Epoch 6965/7000
    1/1 [==============================] - 0s 2ms/step - loss: 0.1898 - binary_accuracy: 1.0000
    Epoch 6966/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1898 - binary_accuracy: 1.0000
    Epoch 6967/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1898 - binary_accuracy: 1.0000
    Epoch 6968/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1897 - binary_accuracy: 1.0000
    Epoch 6969/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1897 - binary_accuracy: 1.0000
    Epoch 6970/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1897 - binary_accuracy: 1.0000
    Epoch 6971/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1897 - binary_accuracy: 1.0000
    Epoch 6972/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1897 - binary_accuracy: 1.0000
    Epoch 6973/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1897 - binary_accuracy: 1.0000
    Epoch 6974/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1896 - binary_accuracy: 1.0000
    Epoch 6975/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1896 - binary_accuracy: 1.0000
    Epoch 6976/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1896 - binary_accuracy: 1.0000
    Epoch 6977/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1896 - binary_accuracy: 1.0000
    Epoch 6978/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1896 - binary_accuracy: 1.0000
    Epoch 6979/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1895 - binary_accuracy: 1.0000
    Epoch 6980/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1895 - binary_accuracy: 1.0000
    Epoch 6981/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1895 - binary_accuracy: 1.0000
    Epoch 6982/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1895 - binary_accuracy: 1.0000
    Epoch 6983/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1895 - binary_accuracy: 1.0000
    Epoch 6984/7000
    1/1 [==============================] - 0s 5ms/step - loss: 0.1894 - binary_accuracy: 1.0000
    Epoch 6985/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1894 - binary_accuracy: 1.0000
    Epoch 6986/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1894 - binary_accuracy: 1.0000
    Epoch 6987/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1894 - binary_accuracy: 1.0000
    Epoch 6988/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1894 - binary_accuracy: 1.0000
    Epoch 6989/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1894 - binary_accuracy: 1.0000
    Epoch 6990/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1893 - binary_accuracy: 1.0000
    Epoch 6991/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1893 - binary_accuracy: 1.0000
    Epoch 6992/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1893 - binary_accuracy: 1.0000
    Epoch 6993/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1893 - binary_accuracy: 1.0000
    Epoch 6994/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1893 - binary_accuracy: 1.0000
    Epoch 6995/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1892 - binary_accuracy: 1.0000
    Epoch 6996/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1892 - binary_accuracy: 1.0000
    Epoch 6997/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1892 - binary_accuracy: 1.0000
    Epoch 6998/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1892 - binary_accuracy: 1.0000
    Epoch 6999/7000
    1/1 [==============================] - 0s 3ms/step - loss: 0.1892 - binary_accuracy: 1.0000
    Epoch 7000/7000
    1/1 [==============================] - 0s 4ms/step - loss: 0.1892 - binary_accuracy: 1.0000
    




    array([[0.01983812],
           [0.18813008],
           [0.18846714],
           [0.72669214]], dtype=float32)



0이 포함된 경우 0.5보다 작은 값을 갖고, 모두가 1인 경우 0.5보다 큰 값을 갖는 것을 확인할 수 있다.


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
# weight
model.layers[0].get_weights()
```




    [array([[2.4401107],
            [2.4379053]], dtype=float32),
     array([-3.900112], dtype=float32)]


