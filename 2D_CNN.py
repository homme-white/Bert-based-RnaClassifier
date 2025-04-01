#!/usr/bin/env python
# coding: utf-8

data_dir = 'data/train'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor


def process_file(file_path, label):
    df = pd.read_csv(file_path, header=None)
    df_new = df.stack().to_frame().T
    return df_new.values.reshape(1, 768, 510, 1), label

# 使用列表来收集数据和标签
df_data = []
df_labels = []

print('Reading positive data ...')
pos_files = glob.glob(os.path.join('data/train/pos', '*.{}'.format('csv')))
with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(executor.map(lambda file_path: process_file(file_path, 1), pos_files))
    for X, y in results:
        df_data.extend(X)
        df_labels.append(y)

print('Reading negative data ...')
neg_files = glob.glob(os.path.join('data/train/neg', '*.{}'.format('csv')))
with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(executor.map(lambda file_path: process_file(file_path, 0), neg_files))
    for X, y in results:
        df_data.extend(X)
        df_labels.append(y)


X_trn = df_data
y_trn = np.array(df_labels)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_trn, y_trn, test_size=0.1 , random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils


nb_classes = 2
nb_epochs = 40

def _2D_CNN_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(1, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(2))

    model.add(Conv2D(128, (3, 3), strides=(1, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), strides=(1, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(2))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


_2D_model = _2D_CNN_model()

# Plot model history
_2D_history = _2D_model.fit(np.asarray(X_train).reshape(len(np.asarray(X_train)),768,510,1), utils.to_categorical(y_train,nb_classes),
                    validation_data=(np.asarray(X_test).reshape(len(np.asarray(X_test)),768,510,1), utils.to_categorical(y_test,nb_classes)),
                    epochs=nb_epochs, batch_size=8, verbose=1)

# 保存模型
model_save_path = '2D_CNN_model.h5'
_2D_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# 加载模型（如果需要）
# _2D_model = tf.keras.models.load_model(model_save_path)

all_y_true = []
all_y_pred = []

# 处理正样本测试集
for fileName in glob.glob(os.path.join('data/test/posa', '*.{}'.format('csv'))):
    df = pd.read_csv(fileName, header=None)
    print('In processing: ', fileName)
    df_new = df.stack().to_frame().T
    X_test = df_new.values.reshape(1, 768, 510, 1)
    y_pred = _2D_model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    all_y_true.append(1)
    all_y_pred.append(y_pred_class[0])

# 处理负样本测试集
for fileName in glob.glob(os.path.join('data/test/nega', '*.{}'.format('csv'))):
    df = pd.read_csv(fileName, header=None)
    print('In processing: ', fileName)
    df_new = df.stack().to_frame().T
    X_test = df_new.values.reshape(1, 768, 510, 1)
    y_pred = _2D_model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    all_y_true.append(0)
    all_y_pred.append(y_pred_class[0])

# 计算整体准确度
accuracy = accuracy_score(all_y_true, all_y_pred)
print(f"Overall accuracy: {accuracy:.4f}")
