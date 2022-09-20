---
title: Phân loại âm thanh (P1)
author: huynhlevu
date: 2022-09-20
categories: [Python,Machine-learning]
tags: [ML, AI]
math: true
mermaid: true
image:
  path: /assets/img/plat.png
  width: 800
  height: 500
  alt: Django - framework for web development.
---
## Cài đặt thư viện

Thư viện hỗ trợ phân tích âm thanh và âm nhạc là Librosa



```
!pip install librosa
!pip install tensorflow
```

## Phân tích dữ liệu



```
import IPython.display as ipd
filepath = "archive/fold1/101415-3-0-2.wav"
ipd.Audio(filepath)
```


 Vì vậy, khi tải bất kỳ tệp âm thanh nào bằng Librosa, nó mang lại 2 điều. Một là tốc độ lấy mẫu và một là mảng hai chiều

 + Tốc độ lấy mẫu - Nó thể hiện số lượng mẫu được ghi lại mỗi giây. Tốc độ lấy mẫu mặc định mà librosa đọc tệp là 22050

 + Mảng 2-D - Trục đầu tiên đại diện cho các biên độ mẫu đã ghi. Và trục thứ hai đại diện cho số lượng kênh. Có nhiều loại kênh khác nhau - Đơn âm (âm thanh có một kênh) và âm thanh nổi (âm thanh có hai kênh).



```
!pip install matplotlib
```





```
import librosa
import librosa.display
import matplotlib.pyplot as plt
data, sample_rate = librosa.load(filepath)
plt.figure(figsize=(12, 5))
librosa.display.waveshow(data, sr=sample_rate)
```



Librosa hiện đang trở nên phổ biến để xử lý tín hiệu âm thanh vì ba lý do sau.

+ Nó cố gắng hội tụ tín hiệu thành mono (một kênh).

+ Nó có thể đại diện cho tín hiệu âm thanh từ -1 đến +1 (ở dạng chuẩn hóa), do đó, một mẫu thông thường được quan sát.

+ Nó cũng có thể xem tốc độ mẫu và theo mặc định, nó chuyển đổi nó thành 22 kHz.

## Kiểm tra tập dữ liệu mất cân bằng

`!pip install pandas`



```
import pandas as pd
metadata = pd.read_csv('csv/UrbanSound8K.csv')
metadata.head(10)
```



sử dụng hàm đếm giá trị để kiểm tra bản ghi của mỗi lớp.

`metadata['class'].value_counts()`

`!pip install seaborn`



```
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.countplot(metadata['class'])
plt.title("Count of records in each class")
plt.xticks(rotation="vertical")
plt.show()
```



## Xử lý trước dữ liệu

Nhiệm vụ là trích xuất một số thông tin quan trọng và giữ dữ liệu ở dạng độc lập (Các tính năng được trích xuất từ ​​tín hiệu âm thanh) và các tính năng phụ thuộc (nhãn lớp). 

Sử dụng Mel Frequency Cepstral coefficients để trích xuất các tính năng độc lập từ tín hiệu âm thanh

MFCC tóm tắt sự phân bố tần số trên kích thước cửa sổ. Vì vậy, có thể phân tích cả đặc tính tần số và thời gian của âm thanh. Biểu diễn âm thanh này sẽ cho phép xác định các đặc điểm để phân loại.



```
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print(mfccs.shape)
print(mfccs)
```



mfccs.shape: https://stackoverflow.com/questions/65206575/what-are-the-components-of-the-mel-mfcc

 ### Trích xuất các tính năng từ tất cả các tệp âm thanh và chuẩn bị khung dữ liệu

```
!pip install numpy
```

res_type: str

Theo mặc định, điều này sử dụng chế độ chất lượng cao của resampy ('kaiser_best').

Để sử dụng một phương pháp nhanh hơn, hãy đặt res_type = 'kaiser_fast'.

Để sử dụng scipy.signal.resample, hãy đặt res_type = 'scipy'.

```
import numpy as np
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # để tìm hiểu các tính năng được chia tỷ lệ, chúng ta sẽ tìm giá trị trung bình của sự chuyển vị của một mảng
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features
```

 thư viện python TQDM để theo dõi tiến trình

```
!pip install tqdm
```

```

import numpy as np
from tqdm import tqdm
import os

extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join('archive/fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])
```

```
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()
```

## Train Test split

Tách tập dữ liệu thành tập dữ liệu độc lập và phụ thuộc



```
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
```



Mã hóa nhãn thành số nguyên



```
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
```



chia dữ liệu thành các tập huấn luyện và thử nghiệm theo tỷ lệ 80-20



```
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
```



## Tạo mô hình phân loại âm thanh

ANN với 3 lớp dày đặc và kiến ​​trúc:

+ Lớp đầu tiên có 100 tế bào thần kinh. Hình dạng đầu vào là 40 theo số lượng tính năng có chức năng kích hoạt là Relu và để tránh bất kỳ trang bị quá mức nào, chúng tôi sẽ sử dụng lớp Dropout với tỷ lệ 0,5.

+ Lớp thứ hai có 200 tế bào thần kinh có chức năng kích hoạt là Relu và lớp Dropout có tỉ lệ là 0,5.

+ Lớp thứ ba lại có 100 tế bào thần kinh với kích hoạt là Relu và lớp Dropout có tỷ lệ là 0,5.



```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

num_labels=y.shape[1]
```



```
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
```

```model.summary()```

## Biên dịch mô hình

Để biên dịch mô hình, chúng ta cần xác định loss function là cross-entropy, accuracy metrics là accuracy score và optimizer là Adam.

```
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
```

## Train model

Đào tạo mô hình và lưu mô hình ở định dạng HDF5. 

Sử dụng callback, đây là một điểm kiểm tra để biết cần bao nhiêu thời gian để đào tạo qua dữ liệu.

Bằng cách đặt verbose = 0, 1 hoặc 2, để 'xem' tiến trình đào tạo cho mỗi kỷ nguyên như thế nào.

verbose=0 sẽ không cho bạn thấy gì (im lặng)

verbose=1 sẽ hiển thị cho bạn một thanh tiến trình progres_bar

verbose=2 sẽ chỉ đề cập đến số kỷ nguyên: Epoch 1/10 ...



```

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 
num_epochs = 100
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath='./audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
```



 (xuất ra các trọng số của mô hình mỗi khi quan sát thấy sự cải thiện trong quá trình đào tạo.)



```python
# checkpoint
# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint] 
```



(Sử dụng EarlyStopping cùng với Checkpoint)



```python
# checkpoint
# from tensorflow.keras.callbacks import EarlyStopping
# filepath="weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# es = EarlyStopping(monitor='val_accuracy', patience=5)
# ->Quá trình huấn luyện này đã dừng lại nếu không có độ chính xác nào đạt được tốt hơn trong năm kỷ nguyên vừa qua
# callbacks_list = [checkpoint, es]
```





```
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)
```



## Kiểm tra độ chính xác của Test



```
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])
```



 sử dụng thuộc tính metrics_names của mô hình để tìm hiểu xem mỗi giá trị trả về của model.evaluate tương ứng với cái gì

`model.metrics_names`

Dự đoán lớp tương ứng cho mỗi tệp âm thanh



```
predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)
print(classes_x)
```



## Kiểm tra một số mẫu âm thanh thử nghiệm

```
filename="archive/fold7/101848-9-0-0.wav"
#preprocess the audio file
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
#Reshape MFCC feature to 2-D array
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
#predicted_label=model.predict_classes(mfccs_scaled_features)
x_predict=model.predict(mfccs_scaled_features) 
predicted_label=np.argmax(x_predict,axis=1)
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label) 
print(prediction_class)
```
