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
model.fit(x, y, epochs=200)
```

    Epoch 1/200
    1/1 [==============================] - 0s 328ms/step - loss: 0.3706 - binary_accuracy: 1.0000
    Epoch 2/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3681 - binary_accuracy: 1.0000
    Epoch 3/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3658 - binary_accuracy: 1.0000
    Epoch 4/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3634 - binary_accuracy: 1.0000
    Epoch 5/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3612 - binary_accuracy: 1.0000
    Epoch 6/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3589 - binary_accuracy: 1.0000
    Epoch 7/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.3568 - binary_accuracy: 1.0000
    Epoch 8/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3546 - binary_accuracy: 1.0000
    Epoch 9/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3525 - binary_accuracy: 1.0000
    Epoch 10/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3505 - binary_accuracy: 1.0000
    Epoch 11/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3485 - binary_accuracy: 1.0000
    Epoch 12/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3465 - binary_accuracy: 1.0000
    Epoch 13/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3446 - binary_accuracy: 1.0000
    Epoch 14/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3427 - binary_accuracy: 1.0000
    Epoch 15/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3409 - binary_accuracy: 1.0000
    Epoch 16/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3391 - binary_accuracy: 1.0000
    Epoch 17/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3373 - binary_accuracy: 1.0000
    Epoch 18/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3355 - binary_accuracy: 1.0000
    Epoch 19/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3338 - binary_accuracy: 1.0000
    Epoch 20/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3321 - binary_accuracy: 1.0000
    Epoch 21/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.3305 - binary_accuracy: 1.0000
    Epoch 22/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3289 - binary_accuracy: 1.0000
    Epoch 23/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.3273 - binary_accuracy: 1.0000
    Epoch 24/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.3257 - binary_accuracy: 1.0000
    Epoch 25/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3241 - binary_accuracy: 1.0000
    Epoch 26/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3226 - binary_accuracy: 1.0000
    Epoch 27/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.3211 - binary_accuracy: 1.0000
    Epoch 28/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3197 - binary_accuracy: 1.0000
    Epoch 29/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3182 - binary_accuracy: 1.0000
    Epoch 30/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3168 - binary_accuracy: 1.0000
    Epoch 31/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3154 - binary_accuracy: 1.0000
    Epoch 32/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.3141 - binary_accuracy: 1.0000
    Epoch 33/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3127 - binary_accuracy: 1.0000
    Epoch 34/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3114 - binary_accuracy: 1.0000
    Epoch 35/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3101 - binary_accuracy: 1.0000
    Epoch 36/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3088 - binary_accuracy: 1.0000
    Epoch 37/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.3075 - binary_accuracy: 1.0000
    Epoch 38/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3063 - binary_accuracy: 1.0000
    Epoch 39/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3050 - binary_accuracy: 1.0000
    Epoch 40/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.3038 - binary_accuracy: 1.0000
    Epoch 41/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3026 - binary_accuracy: 1.0000
    Epoch 42/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.3014 - binary_accuracy: 1.0000
    Epoch 43/200
    1/1 [==============================] - 0s 6ms/step - loss: 0.3003 - binary_accuracy: 1.0000
    Epoch 44/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2991 - binary_accuracy: 1.0000
    Epoch 45/200
    1/1 [==============================] - 0s 6ms/step - loss: 0.2980 - binary_accuracy: 1.0000
    Epoch 46/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2969 - binary_accuracy: 1.0000
    Epoch 47/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2958 - binary_accuracy: 1.0000
    Epoch 48/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2947 - binary_accuracy: 1.0000
    Epoch 49/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2936 - binary_accuracy: 1.0000
    Epoch 50/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2926 - binary_accuracy: 1.0000
    Epoch 51/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2916 - binary_accuracy: 1.0000
    Epoch 52/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2905 - binary_accuracy: 1.0000
    Epoch 53/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2895 - binary_accuracy: 1.0000
    Epoch 54/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2885 - binary_accuracy: 1.0000
    Epoch 55/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2875 - binary_accuracy: 1.0000
    Epoch 56/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2866 - binary_accuracy: 1.0000
    Epoch 57/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2856 - binary_accuracy: 1.0000
    Epoch 58/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2846 - binary_accuracy: 1.0000
    Epoch 59/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2837 - binary_accuracy: 1.0000
    Epoch 60/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2828 - binary_accuracy: 1.0000
    Epoch 61/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2819 - binary_accuracy: 1.0000
    Epoch 62/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2810 - binary_accuracy: 1.0000
    Epoch 63/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2801 - binary_accuracy: 1.0000
    Epoch 64/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2792 - binary_accuracy: 1.0000
    Epoch 65/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2783 - binary_accuracy: 1.0000
    Epoch 66/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2774 - binary_accuracy: 1.0000
    Epoch 67/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2766 - binary_accuracy: 1.0000
    Epoch 68/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2757 - binary_accuracy: 1.0000
    Epoch 69/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2749 - binary_accuracy: 1.0000
    Epoch 70/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2741 - binary_accuracy: 1.0000
    Epoch 71/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2733 - binary_accuracy: 1.0000
    Epoch 72/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2725 - binary_accuracy: 1.0000
    Epoch 73/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2717 - binary_accuracy: 1.0000
    Epoch 74/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2709 - binary_accuracy: 1.0000
    Epoch 75/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2701 - binary_accuracy: 1.0000
    Epoch 76/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2693 - binary_accuracy: 1.0000
    Epoch 77/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2686 - binary_accuracy: 1.0000
    Epoch 78/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2678 - binary_accuracy: 1.0000
    Epoch 79/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2671 - binary_accuracy: 1.0000
    Epoch 80/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2663 - binary_accuracy: 1.0000
    Epoch 81/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2656 - binary_accuracy: 1.0000
    Epoch 82/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2649 - binary_accuracy: 1.0000
    Epoch 83/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2642 - binary_accuracy: 1.0000
    Epoch 84/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2634 - binary_accuracy: 1.0000
    Epoch 85/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2627 - binary_accuracy: 1.0000
    Epoch 86/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2620 - binary_accuracy: 1.0000
    Epoch 87/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2614 - binary_accuracy: 1.0000
    Epoch 88/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2607 - binary_accuracy: 1.0000
    Epoch 89/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2600 - binary_accuracy: 1.0000
    Epoch 90/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2593 - binary_accuracy: 1.0000
    Epoch 91/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2587 - binary_accuracy: 1.0000
    Epoch 92/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2580 - binary_accuracy: 1.0000
    Epoch 93/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2574 - binary_accuracy: 1.0000
    Epoch 94/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2567 - binary_accuracy: 1.0000
    Epoch 95/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2561 - binary_accuracy: 1.0000
    Epoch 96/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2554 - binary_accuracy: 1.0000
    Epoch 97/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2548 - binary_accuracy: 1.0000
    Epoch 98/200
    1/1 [==============================] - 0s 7ms/step - loss: 0.2542 - binary_accuracy: 1.0000
    Epoch 99/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2536 - binary_accuracy: 1.0000
    Epoch 100/200
    1/1 [==============================] - 0s 6ms/step - loss: 0.2530 - binary_accuracy: 1.0000
    Epoch 101/200
    1/1 [==============================] - 0s 8ms/step - loss: 0.2524 - binary_accuracy: 1.0000
    Epoch 102/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2518 - binary_accuracy: 1.0000
    Epoch 103/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2512 - binary_accuracy: 1.0000
    Epoch 104/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2506 - binary_accuracy: 1.0000
    Epoch 105/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2500 - binary_accuracy: 1.0000
    Epoch 106/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2494 - binary_accuracy: 1.0000
    Epoch 107/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2488 - binary_accuracy: 1.0000
    Epoch 108/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2483 - binary_accuracy: 1.0000
    Epoch 109/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2477 - binary_accuracy: 1.0000
    Epoch 110/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2472 - binary_accuracy: 1.0000
    Epoch 111/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2466 - binary_accuracy: 1.0000
    Epoch 112/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2461 - binary_accuracy: 1.0000
    Epoch 113/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2455 - binary_accuracy: 1.0000
    Epoch 114/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2450 - binary_accuracy: 1.0000
    Epoch 115/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2444 - binary_accuracy: 1.0000
    Epoch 116/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2439 - binary_accuracy: 1.0000
    Epoch 117/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2434 - binary_accuracy: 1.0000
    Epoch 118/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2428 - binary_accuracy: 1.0000
    Epoch 119/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2423 - binary_accuracy: 1.0000
    Epoch 120/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2418 - binary_accuracy: 1.0000
    Epoch 121/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2413 - binary_accuracy: 1.0000
    Epoch 122/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2408 - binary_accuracy: 1.0000
    Epoch 123/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2403 - binary_accuracy: 1.0000
    Epoch 124/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2398 - binary_accuracy: 1.0000
    Epoch 125/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2393 - binary_accuracy: 1.0000
    Epoch 126/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2388 - binary_accuracy: 1.0000
    Epoch 127/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2383 - binary_accuracy: 1.0000
    Epoch 128/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2378 - binary_accuracy: 1.0000
    Epoch 129/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2374 - binary_accuracy: 1.0000
    Epoch 130/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2369 - binary_accuracy: 1.0000
    Epoch 131/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2364 - binary_accuracy: 1.0000
    Epoch 132/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2359 - binary_accuracy: 1.0000
    Epoch 133/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2355 - binary_accuracy: 1.0000
    Epoch 134/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2350 - binary_accuracy: 1.0000
    Epoch 135/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2345 - binary_accuracy: 1.0000
    Epoch 136/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2341 - binary_accuracy: 1.0000
    Epoch 137/200
    1/1 [==============================] - 0s 5ms/step - loss: 0.2336 - binary_accuracy: 1.0000
    Epoch 138/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2332 - binary_accuracy: 1.0000
    Epoch 139/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2327 - binary_accuracy: 1.0000
    Epoch 140/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2323 - binary_accuracy: 1.0000
    Epoch 141/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2318 - binary_accuracy: 1.0000
    Epoch 142/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2314 - binary_accuracy: 1.0000
    Epoch 143/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2310 - binary_accuracy: 1.0000
    Epoch 144/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2305 - binary_accuracy: 1.0000
    Epoch 145/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2301 - binary_accuracy: 1.0000
    Epoch 146/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2297 - binary_accuracy: 1.0000
    Epoch 147/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2293 - binary_accuracy: 1.0000
    Epoch 148/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2288 - binary_accuracy: 1.0000
    Epoch 149/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2284 - binary_accuracy: 1.0000
    Epoch 150/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2280 - binary_accuracy: 1.0000
    Epoch 151/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2276 - binary_accuracy: 1.0000
    Epoch 152/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2272 - binary_accuracy: 1.0000
    Epoch 153/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2268 - binary_accuracy: 1.0000
    Epoch 154/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2264 - binary_accuracy: 1.0000
    Epoch 155/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2260 - binary_accuracy: 1.0000
    Epoch 156/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2256 - binary_accuracy: 1.0000
    Epoch 157/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2252 - binary_accuracy: 1.0000
    Epoch 158/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2248 - binary_accuracy: 1.0000
    Epoch 159/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2244 - binary_accuracy: 1.0000
    Epoch 160/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2240 - binary_accuracy: 1.0000
    Epoch 161/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2236 - binary_accuracy: 1.0000
    Epoch 162/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2232 - binary_accuracy: 1.0000
    Epoch 163/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2228 - binary_accuracy: 1.0000
    Epoch 164/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2225 - binary_accuracy: 1.0000
    Epoch 165/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2221 - binary_accuracy: 1.0000
    Epoch 166/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2217 - binary_accuracy: 1.0000
    Epoch 167/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2213 - binary_accuracy: 1.0000
    Epoch 168/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2210 - binary_accuracy: 1.0000
    Epoch 169/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2206 - binary_accuracy: 1.0000
    Epoch 170/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2202 - binary_accuracy: 1.0000
    Epoch 171/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2199 - binary_accuracy: 1.0000
    Epoch 172/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2195 - binary_accuracy: 1.0000
    Epoch 173/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2192 - binary_accuracy: 1.0000
    Epoch 174/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2188 - binary_accuracy: 1.0000
    Epoch 175/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2184 - binary_accuracy: 1.0000
    Epoch 176/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2181 - binary_accuracy: 1.0000
    Epoch 177/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2177 - binary_accuracy: 1.0000
    Epoch 178/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2174 - binary_accuracy: 1.0000
    Epoch 179/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2170 - binary_accuracy: 1.0000
    Epoch 180/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2167 - binary_accuracy: 1.0000
    Epoch 181/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2163 - binary_accuracy: 1.0000
    Epoch 182/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2160 - binary_accuracy: 1.0000
    Epoch 183/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2157 - binary_accuracy: 1.0000
    Epoch 184/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2153 - binary_accuracy: 1.0000
    Epoch 185/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2150 - binary_accuracy: 1.0000
    Epoch 186/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2147 - binary_accuracy: 1.0000
    Epoch 187/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2143 - binary_accuracy: 1.0000
    Epoch 188/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2140 - binary_accuracy: 1.0000
    Epoch 189/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2137 - binary_accuracy: 1.0000
    Epoch 190/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2133 - binary_accuracy: 1.0000
    Epoch 191/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2130 - binary_accuracy: 1.0000
    Epoch 192/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2127 - binary_accuracy: 1.0000
    Epoch 193/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2124 - binary_accuracy: 1.0000
    Epoch 194/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2121 - binary_accuracy: 1.0000
    Epoch 195/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2117 - binary_accuracy: 1.0000
    Epoch 196/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2114 - binary_accuracy: 1.0000
    Epoch 197/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2111 - binary_accuracy: 1.0000
    Epoch 198/200
    1/1 [==============================] - 0s 4ms/step - loss: 0.2108 - binary_accuracy: 1.0000
    Epoch 199/200
    1/1 [==============================] - 0s 3ms/step - loss: 0.2105 - binary_accuracy: 1.0000
    Epoch 200/200
    1/1 [==============================] - 0s 2ms/step - loss: 0.2102 - binary_accuracy: 1.0000
    




    <keras.callbacks.History at 0x235a2a18af0>




```python
model.predict([-5, -3.9, -2, -1.4, -0.89, 0.25, 0.38, 1.7, 3.5, 4.6])
```




    array([[0.00952029],
           [0.02634037],
           [0.1391241 ],
           [0.22129261],
           [0.31466907],
           [0.57298607],
           [0.60260683],
           [0.83998203],
           [0.96615064],
           [0.987705  ]], dtype=float32)



음수는 0.5보다 작은 값을 갖고, 양수는 0.5보다 큰 값을 갖는 것을 확인할 수 있다.


```python
model.predict([-1000, 1000])
```




    array([[0.],
           [1.]], dtype=float32)



극한적인 값의 결과 0과 1이 된다.


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
# weight
model.layers[0].get_weights()
```




    [array([[0.94072384]], dtype=float32), array([0.05886382], dtype=float32)]


