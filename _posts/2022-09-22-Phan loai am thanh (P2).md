---
title: Phân loại âm thanh (P2)
author: huynhlevu
date: 2022-09-22
categories: [Python,Machine-learning]
tags: [ML, AI]
math: true
mermaid: true
image:
  path: /assets/img/plat2.jpg
  width: 800
  height: 500
  alt: Phân loại âm thanh P2.
---

## Khái niệm cơ bản về phân loại âm thanh

### Dạng sóng (Waveform)

Âm thanh là những dao động do một vật tạo ra khi các phần tử không khí xung quanh dao động. Âm thanh là một sóng cơ học, nơi năng lượng được truyền từ nguồn này sang nguồn khác. Dạng sóng là một biểu diễn giản đồ giúp chúng ta phân tích sự dịch chuyển của sóng âm thanh theo thời gian, cùng với một số tham số thiết yếu khác cần thiết cho một nhiệm vụ cụ thể.

Mặt khác, tần số ở dạng sóng là đại diện của số lần một dạng sóng lặp lại chính nó trong khoảng thời gian một giây. Đỉnh của dạng sóng ở trên cùng được gọi là đỉnh, trong khi điểm dưới cùng được gọi là đáy. Biên độ là khoảng cách từ đường tâm đến đỉnh của máng hoặc đáy của đỉnh.

### Quang phổ (Spectrograms)

Quang phổ là một biểu diễn trực quan của phổ tần số của một tín hiệu khi nó thay đổi theo thời gian. Khi được áp dụng cho một tín hiệu âm thanh , các bản ghi quang phổ đôi khi được gọi là bản ghi âm

## Dự án Phân loại Âm thanh

Bằng cách chuyển đổi dạng sóng thô của dữ liệu âm thanh thành dạng quang phổ, chúng ta có thể chuyển nó qua các mô hình học sâu để giải thích và phân tích dữ liệu. 

Trong dự án này, mục tiêu là thu lại âm thanh phát ra từ một con chim. Khi đã thu được thành công dạng sóng, chúng ta có thể tiến hành chuyển dạng sóng này thành một biểu đồ quang phổ, đây là biểu diễn trực quan của dạng sóng có sẵn. Vì những hình ảnh phổ này là hình ảnh trực quan, chúng ta có thể sử dụng mạng nơ-ron tích tụ để phân tích chúng cho phù hợp bằng cách tạo mô hình học sâu để tính toán kết quả phân loại nhị phân

## Thực hiện dự án Nhận dạng và Phân loại Âm thanh với Học sâu:

Mục tiêu của dự án là đọc âm thanh đến từ một khu rừng và giải thích xem dữ liệu nhận được thuộc về một loài chim cụ thể (chim mũ lưỡi trai) hay là một số tiếng ồn khác mà ta không thực sự quan tâm đến việc ghi nhận.

Thư viện TensorFlow-io, sẽ cấp cho ta quyền truy cập vào các hệ thống tệp và định dạng tệp không có sẵn trong hỗ trợ tích hợp của TensorFlow

```python
!pip install tensorflow-io[tensorflow]
```

Import các thư viện cần thiết

```python
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import tensorflow_io as tfio
from matplotlib import pyplot as plt
import os
```

Tải dataset: https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing

Có ba thư mục trong thư mục data. Ba thư mục cụ thể là các bản ghi âm trong rừng chứa một đoạn clip dài ba phút về âm thanh được tạo ra trong rừng, đoạn clip dài ba giây về các bản ghi âm của chim Capuchin và đoạn ghi âm dài ba giây về âm thanh không do chim Capuchin tạo ra

```python
CAPUCHIN_FILE = os.path.join('data', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')
```

 Hàm được xác định trong đoạn mã dưới đây sẽ cho phép ta đọc dữ liệu và chuyển đổi nó thành một kênh đơn (hoặc đơn) để phân tích dễ dàng hơn

Hàm tf.squeeze() trả về một tensor có cùng giá trị với đối số đầu tiên của nó, nhưng có hình dạng khác. Nó loại bỏ các thứ nguyên có kích thước là một

```python
def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav
```

### Chuẩn bị tập dữ liệu

Nhãn 1: tiếng chim Capuchin

Nhãn 0: tín hiệu âm thanh nhiễu

tf.data.Dataset.list_file: để quét tất cả các phần tử thư mục data

```python
# Defining the positive and negative paths
POS = os.path.join('data', 'Parsed_Capuchinbird_Clips/*.wav')
NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips/*.wav')

# Creating the Datasets
pos = tf.data.Dataset.list_files(POS)
neg = tf.data.Dataset.list_files(NEG)

# Adding labels
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)
```

Phân tích bước sóng trung bình của chim Capuchin

```python
# Analyzing the average wavelength of a Capuchin bird
lengths = []
for file in os.listdir(os.path.join('data', 'Parsed_Capuchinbird_Clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('data', 'Parsed_Capuchinbird_Clips', file))
    lengths.append(len(tensor_wave))
```

```python
from statistics import mean
mean(lengths)
```

```python
wav = load_wav_16k_mono('data/Parsed_Capuchinbird_Clips/XC22397-2.wav')
wav = wav[:48000]
zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
wav = tf.concat([zero_padding, wav],0)
spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
spectrogram = tf.abs(spectrogram)

spectrogram = tf.expand_dims(spectrogram, axis=2)
```

Thu thập tất cả các dạng sóng và tính toán Tín hiệu Biến đổi Fourier trong thời gian ngắn với thư viện TensorFlow để có được một biểu diễn trực quan

```python
def preprocess(file_path, label): 
        wav = load_wav_16k_mono(file_path)
        wav = wav[:48000]
        zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav],0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)

        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram, label

 
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

spectrogram, label = preprocess(filepath, label)
```

### Xây dựng mô hình học sâu

Tải các phần tử dữ liệu chương trình quang phổ thu được từ chức năng bước tiền xử lý và lưu vào bộ nhớ cache và xáo trộn dữ liệu này bằng cách sử dụng các chức năng có sẵn của TensorFlow, cũng như tạo kích thước lô gồm mười sáu để tải các phần tử dữ liệu cho phù hợp.

tf.data.Dataset.map(): được sử dụng để chuyển đổi các mục trong tập dữ liệu

Sử dụng tf.data.Dataset.cache()thực sự không phải là một lựa chọn tốt vì nó sẽ lưu toàn bộ tập dữ liệu vào bộ nhớ, điều này gây mất thời gian và có thể làm tràn bộ nhớ của bạn

shuffle( buffer_size, seed=None, reshuffle_each_iteration=None) Phương pháp xáo trộn các mẫu trong tập dữ liệu. Buffer_size là số lượng mẫu được lấy ngẫu nhiên và trả về dưới dạng tf.Dataset.

batch(batch_size,drop_remainder=False)Tạo các lô của tập dữ liệu với kích thước lô được cung cấp vì batch_size cũng là độ dài của các lô.

Ngay sau khi tất cả các mục được đọc từ tập dữ liệu và bạn cố đọc phần tử tiếp theo, tập dữ liệu sẽ gặp lỗi. Đó là nơi ds.repeat() phát huy tác dụng. Nó sẽ khởi tạo lại tập dữ liệu

Hầu hết các đường ống đầu vào của tập dữ liệu phải kết thúc bằng một lệnh gọi đến prefetch. Điều này cho phép các phần tử sau này được chuẩn bị trong khi phần tử hiện tại đang được xử lý. Điều này thường cải thiện độ trễ và thông lượng, với chi phí sử dụng bộ nhớ bổ sung để lưu trữ các phần tử tìm nạp trước.

```python
# Creating a Tensorflow Data Pipeline
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)
```

Trước khi tiến hành xây dựng mô hình học sâu, ta có thể tạo phân vùng cho các mẫu thử nghiệm và đào tạo, như được hiển thị trong đoạn mã bên dưới.

```python
# Split into Training and Testing Partitions
train = data.take(36)
test = data.skip(36).take(15)
```

```python
# độ dài của tập dữ liệu
tf.data.experimental.cardinality(train)
```

```python
# hiển thị shape của tf.Tensor
for i in train:
    print(i)
```

Xây dựng mô hình kiểu tuần tự

Xây dựng hai khối lớp chập với mười sáu bộ lọc và kích thước hạt nhân là (3, 3)

Chức năng kích hoạt ReLU được sử dụng trong việc xây dựng các lớp tích tụ. 

Sau đó, chúng ta có thể tiến hành làm phẳng đầu ra thu được từ các lớp tích tụ để làm cho nó phù hợp cho quá trình xử lý tiếp theo. 

Cuối cùng, chúng ta có thể thêm các lớp được kết nối đầy đủ với chức năng kích hoạt Sigmoid với một nút đầu ra để nhận đầu ra phân loại nhị phân. 

```python
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
# model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
```

Để biên dịch mô hình, chúng ta có thể sử dụng trình tối ưu hóa Adam, hàm mất mát entropy chéo nhị phân để phân loại nhị phân và xác định một số số liệu thu hồi và độ chính xác bổ sung cho phân tích mô hình.

```python
# Compiling and fitting the model
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

model.fit(train, epochs=4, validation_data=test)
```

```python
model.save('mymodel.h5')
```

```python
from keras.models import load_model
model = load_model('mymodel.h5')
```

### Đưa ra các Dự đoán

```python
# Prediction for a single batch
X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)

# converting logits to classes
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
```

Hàm bên dưới nhận đầu vào định dạng mp3 và chuyển đổi chúng thành tensor. Sau đó, ta tính giá trị trung bình của đầu vào đa kênh để chuyển nó thành kênh đơn và thu được tín hiệu tần số mong muốn.

```python
def load_mp3_16k_mono(filename):
    """ Load an audio file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
    # Convert to tensor and combine channels 
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    # Extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav
    
mp3 = os.path.join('data', 'Forest Recordings', 'recording_00.mp3')

wav = load_mp3_16k_mono(mp3)

audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)

samples, index = audio_slices.as_numpy_iterator().next()
```

Ta sẽ xây dựng một hàm giúp tách các phân đoạn riêng lẻ thành các bản đồ quang phổ có cửa sổ để tính toán thêm. Ta sẽ ánh xạ dữ liệu phù hợp và tạo các lát cắt thích hợp để đưa ra các dự đoán cần thiết

```python
# Build Function to Convert Clips into Windowed Spectrograms
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram
```

```python
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
```

```python
#audio_slices = audio_slices.batch(64)
audio_slices = audio_slices.batch(32)
yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
```

Ta sẽ chạy quy trình sau cho tất cả các tệp trong bản ghi rừng và thu được tổng kết quả được tính toán. Kết quả sẽ chứa các đoạn số không và các đoạn trong đó tổng số các số đó được xuất ra để tính điểm tổng thể của các đoạn clip. Chúng ta có thể tìm ra tổng số tiếng kêu của chim Capuchin trong bản ghi âm theo yêu cầu của dự án

```python
results = {}
class_preds = {}

for file in os.listdir(os.path.join('data', 'Forest Recordings')):
    FILEPATH = os.path.join('data','Forest Recordings', file)
    
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    yhat = model.predict(audio_slices)
    
    results[file] = yhat
    
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
class_preds
```
