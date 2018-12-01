# duck_pattern-recongnition
#107年_東華大學資工系所_圖樣辨識課程_作業_1_辨識鴨子 教授：江師政欽 學生：陳建宏
#養鴨場空拍照(像素 9555 * 3180 * 彩圖)
#read images for train
#經分析養鴨場的圖片內容，將label分類為5類：
#1、鴨；2、砂地；3、池水；4、草；5、全黑
#自行切劃每類代表的圖(像素 35 * 30 * 彩圖)
#每類的圖片張數皆為26張，總共130張供訓練用
import cv2
import numpy as np
from pandas import read_csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

img_all = []

i = 1
for i in range(1, 131):
    img_all.append(cv2.imread(str(i) + '.jpg'))
img = np.array(img_all, 'float32')
img.shape

#正規化
img_normalize = img / 255

# 定義 label 

label = read_csv('label.csv')
label_array = np.array(label, 'int')
label_array.shape
label = np_utils.to_categorical(label_array)

# 搞定模型
np.random.seed(10)
model = Sequential()
model.add(Conv2D(filters = 36, kernel_size = (5, 5), padding = 'same', input_shape = (35, 30, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 36, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(5, activation = 'softmax'))
print(model.summary())

# 開始訓練模型
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
train_history =model.fit(x = img_normalize, y = label, validation_split = 0.1, epochs = 6, batch_size = 1, verbose = 2)
# test datas
#read image
img_test = cv2.imread('full_duck_9555_3180.jpg')
img_array = np.array(img_test, 'float32')
img_array.shape
img_reshape = img_array.reshape(28938, 35, 30, 3)
img_reshape.shape
img_test_normalize = img_reshape / 255
prediction = model.predict_classes(img_test_normalize)
prediction.shape
prediction_duck = prediction.reshape(273, 106, 1)
prediction_duck.shape
