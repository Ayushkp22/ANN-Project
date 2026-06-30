import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
     
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

     

df=  pd.read_csv('Iris.csv')
     

df.head()
     
Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
0	1	5.1	3.5	1.4	0.2	Iris-setosa
1	2	4.9	3.0	1.4	0.2	Iris-setosa
2	3	4.7	3.2	1.3	0.2	Iris-setosa
3	4	4.6	3.1	1.5	0.2	Iris-setosa
4	5	5.0	3.6	1.4	0.2	Iris-setosa

df['Species'].value_counts()
     
count
Species	
Iris-setosa	50
Iris-versicolor	50
Iris-virginica	50

dtype: int64

df.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 6 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             150 non-null    int64  
 1   SepalLengthCm  150 non-null    float64
 2   SepalWidthCm   150 non-null    float64
 3   PetalLengthCm  150 non-null    float64
 4   PetalWidthCm   150 non-null    float64
 5   Species        150 non-null    object 
dtypes: float64(4), int64(1), object(1)
memory usage: 7.2+ KB

sns.pairplot(df,hue ='Species')
     
<seaborn.axisgrid.PairGrid at 0x78fbb127f8f0>


X = df.drop(columns= ['Species','Id'],axis = 1 )
y = df['Species']
     

encoder = LabelEncoder()
y_int = encoder.fit_transform(y)
     

y_int
     
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


     

X_train, X_test, y_train, y_test = train_test_split(
...     X, y_int, test_size=0.2, random_state=42,stratify=y_int)
     

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
     

per = Perceptron(max_iter = 1000,random_state=42)
per.fit(X_train_scaled, y_train)
     
Perceptron(random_state=42)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

y_pred_percep = per.predict(X_test_scaled)
     

accuracy_score(y_test,y_pred_percep)
     
0.8666666666666667

print(classification_report(y_test,y_pred_percep))
     
              precision    recall  f1-score   support

           0       0.83      1.00      0.91        10
           1       0.88      0.70      0.78        10
           2       0.90      0.90      0.90        10

    accuracy                           0.87        30
   macro avg       0.87      0.87      0.86        30
weighted avg       0.87      0.87      0.86        30


y_train_cat = to_categorical(y_train,num_classes = 3)
y_test_cat = to_categorical(y_test,num_classes = 3)

     


     

model = Sequential([
    Dense(16,input_dim = 4,activation='relu'),
    Dense(8,activation='relu'),
    Dense(3,activation='softmax')
])
     

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
     

history = model.fit(X_train_scaled,y_train_cat,
                    epochs = 100,batch_size= 8, validation_split = 0.2,verbose = 1)
     
Epoch 1/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - accuracy: 0.2798 - loss: 1.2381 - val_accuracy: 0.4583 - val_loss: 1.1051
Epoch 2/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.4022 - loss: 1.1474 - val_accuracy: 0.5417 - val_loss: 1.0498
Epoch 3/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.6091 - loss: 1.0177 - val_accuracy: 0.6250 - val_loss: 0.9909
Epoch 4/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.6249 - loss: 1.0290 - val_accuracy: 0.7500 - val_loss: 0.9290
Epoch 5/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.6921 - loss: 0.9298 - val_accuracy: 0.7917 - val_loss: 0.8676
Epoch 6/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.7498 - loss: 0.8600 - val_accuracy: 0.7917 - val_loss: 0.8107
Epoch 7/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.7436 - loss: 0.8139 - val_accuracy: 0.7500 - val_loss: 0.7576
Epoch 8/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.7557 - loss: 0.7447 - val_accuracy: 0.7500 - val_loss: 0.7112
Epoch 9/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.7725 - loss: 0.6687 - val_accuracy: 0.7500 - val_loss: 0.6700
Epoch 10/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.7699 - loss: 0.6146 - val_accuracy: 0.8333 - val_loss: 0.6344
Epoch 11/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.7923 - loss: 0.6073 - val_accuracy: 0.8333 - val_loss: 0.6045
Epoch 12/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8155 - loss: 0.5844 - val_accuracy: 0.8333 - val_loss: 0.5763
Epoch 13/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8627 - loss: 0.5105 - val_accuracy: 0.8333 - val_loss: 0.5520
Epoch 14/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8265 - loss: 0.5148 - val_accuracy: 0.8333 - val_loss: 0.5301
Epoch 15/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.7852 - loss: 0.5132 - val_accuracy: 0.8333 - val_loss: 0.5123
Epoch 16/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8431 - loss: 0.4158 - val_accuracy: 0.8333 - val_loss: 0.4965
Epoch 17/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.7961 - loss: 0.4367 - val_accuracy: 0.8333 - val_loss: 0.4832
Epoch 18/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.7824 - loss: 0.4402 - val_accuracy: 0.8333 - val_loss: 0.4714
Epoch 19/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8255 - loss: 0.4354 - val_accuracy: 0.8333 - val_loss: 0.4589
Epoch 20/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8033 - loss: 0.4076 - val_accuracy: 0.8333 - val_loss: 0.4493
Epoch 21/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8219 - loss: 0.4186 - val_accuracy: 0.8333 - val_loss: 0.4412
Epoch 22/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.8374 - loss: 0.3567 - val_accuracy: 0.8333 - val_loss: 0.4338
Epoch 23/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8953 - loss: 0.3213 - val_accuracy: 0.8333 - val_loss: 0.4272
Epoch 24/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.8464 - loss: 0.3472 - val_accuracy: 0.8333 - val_loss: 0.4163
Epoch 25/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.8365 - loss: 0.3321 - val_accuracy: 0.8333 - val_loss: 0.4025
Epoch 26/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.7913 - loss: 0.3817 - val_accuracy: 0.8333 - val_loss: 0.3968
Epoch 27/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8257 - loss: 0.3510 - val_accuracy: 0.8333 - val_loss: 0.3900
Epoch 28/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8583 - loss: 0.3116 - val_accuracy: 0.8333 - val_loss: 0.3822
Epoch 29/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8970 - loss: 0.2912 - val_accuracy: 0.8333 - val_loss: 0.3764
Epoch 30/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.8615 - loss: 0.2785 - val_accuracy: 0.8333 - val_loss: 0.3682
Epoch 31/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.8134 - loss: 0.3485 - val_accuracy: 0.8333 - val_loss: 0.3579
Epoch 32/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - accuracy: 0.8520 - loss: 0.3075 - val_accuracy: 0.8333 - val_loss: 0.3529
Epoch 33/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.8777 - loss: 0.2415 - val_accuracy: 0.8333 - val_loss: 0.3511
Epoch 34/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.8885 - loss: 0.2672 - val_accuracy: 0.8333 - val_loss: 0.3382
Epoch 35/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.7953 - loss: 0.3493 - val_accuracy: 0.8333 - val_loss: 0.3317
Epoch 36/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step - accuracy: 0.8622 - loss: 0.2714 - val_accuracy: 0.8333 - val_loss: 0.3270
Epoch 37/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step - accuracy: 0.8377 - loss: 0.3212 - val_accuracy: 0.8333 - val_loss: 0.3202
Epoch 38/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.9113 - loss: 0.2215 - val_accuracy: 0.8333 - val_loss: 0.3135
Epoch 39/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - accuracy: 0.8950 - loss: 0.2503 - val_accuracy: 0.8333 - val_loss: 0.3078
Epoch 40/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.8712 - loss: 0.2518 - val_accuracy: 0.8333 - val_loss: 0.2962
Epoch 41/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8359 - loss: 0.2878 - val_accuracy: 0.8333 - val_loss: 0.2904
Epoch 42/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8294 - loss: 0.3089 - val_accuracy: 0.8333 - val_loss: 0.2841
Epoch 43/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8890 - loss: 0.2690 - val_accuracy: 0.8333 - val_loss: 0.2814
Epoch 44/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8783 - loss: 0.2522 - val_accuracy: 0.8333 - val_loss: 0.2740
Epoch 45/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8711 - loss: 0.2654 - val_accuracy: 0.8333 - val_loss: 0.2695
Epoch 46/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8390 - loss: 0.2671 - val_accuracy: 0.8750 - val_loss: 0.2589
Epoch 47/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.8671 - loss: 0.2346 - val_accuracy: 0.8750 - val_loss: 0.2524
Epoch 48/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8913 - loss: 0.2400 - val_accuracy: 0.8750 - val_loss: 0.2479
Epoch 49/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9103 - loss: 0.1890 - val_accuracy: 0.8750 - val_loss: 0.2427
Epoch 50/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9097 - loss: 0.2011 - val_accuracy: 0.9167 - val_loss: 0.2368
Epoch 51/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8881 - loss: 0.2107 - val_accuracy: 0.9167 - val_loss: 0.2247
Epoch 52/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.8717 - loss: 0.2252 - val_accuracy: 0.9167 - val_loss: 0.2221
Epoch 53/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9034 - loss: 0.2233 - val_accuracy: 0.9167 - val_loss: 0.2190
Epoch 54/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9046 - loss: 0.2178 - val_accuracy: 0.9167 - val_loss: 0.2069
Epoch 55/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9106 - loss: 0.2199 - val_accuracy: 0.9167 - val_loss: 0.2005
Epoch 56/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8854 - loss: 0.2339 - val_accuracy: 0.9583 - val_loss: 0.1946
Epoch 57/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9161 - loss: 0.2013 - val_accuracy: 0.9583 - val_loss: 0.1902
Epoch 58/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9153 - loss: 0.1905 - val_accuracy: 0.9583 - val_loss: 0.1845
Epoch 59/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9319 - loss: 0.1668 - val_accuracy: 0.9583 - val_loss: 0.1767
Epoch 60/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9532 - loss: 0.1452 - val_accuracy: 0.9583 - val_loss: 0.1698
Epoch 61/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9249 - loss: 0.1802 - val_accuracy: 0.9583 - val_loss: 0.1561
Epoch 62/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9313 - loss: 0.1750 - val_accuracy: 0.9583 - val_loss: 0.1499
Epoch 63/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9177 - loss: 0.1649 - val_accuracy: 0.9583 - val_loss: 0.1442
Epoch 64/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9233 - loss: 0.2127 - val_accuracy: 1.0000 - val_loss: 0.1378
Epoch 65/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9504 - loss: 0.1478 - val_accuracy: 1.0000 - val_loss: 0.1335
Epoch 66/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9314 - loss: 0.1879 - val_accuracy: 1.0000 - val_loss: 0.1243
Epoch 67/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9587 - loss: 0.1251 - val_accuracy: 1.0000 - val_loss: 0.1220
Epoch 68/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9264 - loss: 0.1649 - val_accuracy: 1.0000 - val_loss: 0.1177
Epoch 69/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.9367 - loss: 0.1524 - val_accuracy: 1.0000 - val_loss: 0.1092
Epoch 70/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9704 - loss: 0.1258 - val_accuracy: 1.0000 - val_loss: 0.1030
Epoch 71/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9857 - loss: 0.1198 - val_accuracy: 1.0000 - val_loss: 0.0987
Epoch 72/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9743 - loss: 0.1168 - val_accuracy: 1.0000 - val_loss: 0.0967
Epoch 73/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9558 - loss: 0.1167 - val_accuracy: 1.0000 - val_loss: 0.0929
Epoch 74/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9651 - loss: 0.1168 - val_accuracy: 1.0000 - val_loss: 0.0863
Epoch 75/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9351 - loss: 0.1188 - val_accuracy: 1.0000 - val_loss: 0.0850
Epoch 76/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9295 - loss: 0.1365 - val_accuracy: 1.0000 - val_loss: 0.0807
Epoch 77/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9652 - loss: 0.1074 - val_accuracy: 1.0000 - val_loss: 0.0769
Epoch 78/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9799 - loss: 0.1014 - val_accuracy: 1.0000 - val_loss: 0.0731
Epoch 79/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9567 - loss: 0.1056 - val_accuracy: 1.0000 - val_loss: 0.0702
Epoch 80/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9663 - loss: 0.0949 - val_accuracy: 1.0000 - val_loss: 0.0674
Epoch 81/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9452 - loss: 0.1193 - val_accuracy: 1.0000 - val_loss: 0.0631
Epoch 82/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9143 - loss: 0.1439 - val_accuracy: 1.0000 - val_loss: 0.0628
Epoch 83/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9463 - loss: 0.0966 - val_accuracy: 1.0000 - val_loss: 0.0604
Epoch 84/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9581 - loss: 0.1022 - val_accuracy: 1.0000 - val_loss: 0.0562
Epoch 85/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9296 - loss: 0.1357 - val_accuracy: 1.0000 - val_loss: 0.0540
Epoch 86/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9819 - loss: 0.0717 - val_accuracy: 1.0000 - val_loss: 0.0534
Epoch 87/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9827 - loss: 0.0802 - val_accuracy: 1.0000 - val_loss: 0.0510
Epoch 88/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9665 - loss: 0.0900 - val_accuracy: 1.0000 - val_loss: 0.0486
Epoch 89/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9106 - loss: 0.1308 - val_accuracy: 1.0000 - val_loss: 0.0475
Epoch 90/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9753 - loss: 0.0761 - val_accuracy: 1.0000 - val_loss: 0.0457
Epoch 91/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9330 - loss: 0.1043 - val_accuracy: 1.0000 - val_loss: 0.0442
Epoch 92/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9678 - loss: 0.0742 - val_accuracy: 1.0000 - val_loss: 0.0421
Epoch 93/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9584 - loss: 0.0773 - val_accuracy: 1.0000 - val_loss: 0.0407
Epoch 94/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9733 - loss: 0.0642 - val_accuracy: 1.0000 - val_loss: 0.0408
Epoch 95/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9538 - loss: 0.0885 - val_accuracy: 1.0000 - val_loss: 0.0390
Epoch 96/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9430 - loss: 0.0885 - val_accuracy: 1.0000 - val_loss: 0.0377
Epoch 97/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9509 - loss: 0.0845 - val_accuracy: 1.0000 - val_loss: 0.0365
Epoch 98/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9413 - loss: 0.0860 - val_accuracy: 1.0000 - val_loss: 0.0353
Epoch 99/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.9820 - loss: 0.0606 - val_accuracy: 1.0000 - val_loss: 0.0337
Epoch 100/100
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.9802 - loss: 0.0843 - val_accuracy: 1.0000 - val_loss: 0.0331

loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=1)
print(acc)
     
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - accuracy: 0.9333 - loss: 0.1194
0.9333333373069763

plt.figure(figsize = (10,4))
plt.plot(history.history['accuracy'],label = "train Acc")
plt.plot(history.history['val_accuracy'],label = "val Acc")


     
[<matplotlib.lines.Line2D at 0x78fb8e7bef00>]



     
