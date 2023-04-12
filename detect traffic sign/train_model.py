import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten

data = []
label = []
classes = 3
cur_path = os.getcwd()  # lấy đường dẫn hiện tại đang làm việc

# đọc ảnh và gắn nhãn cho chúng
for j in range(classes):
    path = os.path.join(cur_path, str(j))
    number_image = os.listdir(path)

    for p in range(len(number_image)):
        try:
            img = cv2.imread(os.path.join(path, str(p) + '.jpg'))
            pixel = cv2.resize(img, (64, 64))
            data.append(np.array(pixel))
            label.append(j)
        except:
            print(j)
            print(p)
            print('Error loading image')

data = np.array(data)
label = np.array(label)

# chia dữ liệu cho train và val
x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.2, random_state=42)

# chuyển thành one-hot encoding
y_train = np_utils.to_categorical(y=y_train, num_classes=3)
y_val = np_utils.to_categorical(y=y_val, num_classes=3)

input_shape = (64, 64, 3)

# khởi tạo model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(3, activation='softmax'))


# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# train model
epochs = 8
batch_size = 16
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
model.save('traffic_sign.h5')