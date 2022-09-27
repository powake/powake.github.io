---
title: Tổng quan về xử lý ngôn ngữ tự nhiên
author: huynhlevu
date: 2022-09-26
categories: [Python,Machine-learning]
tags: [ML, AI]
math: true
mermaid: true
image:
  path: /assets/img/NLP.jpeg
  width: 800
  height: 500
  alt: Tổng quan về xử lý ngôn ngữ tự nhiên
---

# <font color = 'red'> I. Giới thiệu về Xử lý ngôn ngữ tự nhiên (Natural Language Proccessing)

Xử lý ngôn ngữ tự nhiên cung cấp một bộ công cụ và thuật toán rất cần thiết để hiểu và xử lý khối lượng lớn dữ liệu phi cấu trúc trong thế giới hiện nay. Hiện nay Deep learning đã được áp dụng rộng rãi cho nhiều nhiệm vụ xử lý ngôn ngữ tự nhiên vì hiệu suất deep learning hiệu quả đáng kể trong các nhiệm vụ như phân loại hình ảnh, nhận dạng giọng nói và tạo văn bản thực tế,... Tensorflow là một deep learning framework trực quan và hiệu quả nhất cho các nhiệm vụ này.

## <font color = 'blue'> 1.Xử lý ngôn ngữ tự nhiên là gì

Mục tiêu của xử lý ngôn ngữ tự nhiên là làm cho máy móc hiểu các ngôn ngữ nói và viết của chúng ta. Xử lý ngôn ngữ tự nhiên có mặt ở khắp nơi và đã là một phần lớn trong cuộc sống của con người. Trợ lý ảo (Virtual Assistants) chẳng hạn như Google Assistant, Cortana, Alexa và Apple Siri,.. phần lớn là các hệ thống NLP.

NLP là một lĩnh vực nghiên cứu cực kỳ thách thức vì các từ và ngữ nghĩa có mối quan hệ phi tuyến tính rất phức tạp. Vấn đề còn khó khăn hơn khi mỗi ngôn ngữ có ngữ pháp, cú pháp và từ vựng riêng. Do đó, xử lý dữ liệu văn bản liên quan đến các nhiệm vụ phức tạo khác nhau như phân tích cú pháp văn bản, hiểu cấu trúc ngữ pháp cơ bản của ngôn ngữ. Ví dụ trong hai câu này,"tôi đi trên đường" và "tôi thêm đường vào nước cam", từ "đường" có hai ý nghĩa hoàn toàn khác nhau, do bối cảnh mà nó được sử dụng.Để hiểu bối cảnh mà từ đang được sử dụng. Học máy đã trở thành một yếu tố chính cho NLP, giúp hoàn thành các nhiệm vụ đã nói ở trên thông qua học máy.

## <font color ='blue'> 2. Nhiệm vụ của xử lý ngôn ngữ tự nhiên

- **Tokenization**: là nhiệm vụ tách một kho văn bản thành các đơn vị nhỏ hơn (ví dụ: từ hoặc ký tự). Mặc dù nó có vẻ bình thường đối với một số ngôn ngữ như tiếng Anh nhưng lại khó khăn đối với các ngôn ngữ như tiếng Nhật vì các từ không được phân định bởi khoảng cách hay dấu chấm câu
- **Word-Sense Disambiguation**: là nhiệm vụ xác định đúng ý nghĩa của từ
- **Nhận dạng thực thể (Named Entity Recognition(NER))**: cố gắng trích xuất các thực thể (ví dụ: người, vị trí, tổ chức) từ một phần nhất định của văn bản. Ví dụ, mẫu câu , "John gave Mary two apples at school on Monday" sẽ được chuyển đổi thành "[John]name gave [Mary]name [two]number apples at [school]organization on [Monday]time". NER là một chủ đề bắt buộc trong các lĩnh vực như truy xuất thông tin.
- **Gán nhãn từ loại (Part-of-Speech tagging POS)**: là nhiệm vụ gán các từ loại cho các phần của văn bản. Nó có thể là các thẻ danh từ, động từ, tính từ, trạng từ và giới từ,...
- **Phân loại câu/tóm tắt văn bản**: Phân loại câu hoặc bản tóm tắt (ví dụ, đánh giá phim) có nhiều trường hợp sử dụng như phát hiện thư rác, phân loại bài báo (ví dụ: chính trị, công nghệ và thể thao) và xếp hạng đánh giá sản phẩm (nghĩa là tích cực hoặc tiêu cực). Điều này đạt được bằng cách đào tạo một mô hình phân loại với dữ liệu được dán nhãn (nghĩa là đánh giá được chú thích bởi con người, với nhãn tích cực hoặc tiêu cực).
- **Tạo văn bản**: Trong tạo văn bản, một mô hình học tập (ví dụ: mạng thần kinh) được đào tạo bằng văn bản (một bộ sưu tập lớn các tài liệu văn bản) và sau đó nó dự đoán văn bản mới.Ví dụ, mô hình ngôn ngữ có thể xuất hiện một câu chuyện khoa học viễn tưởng hoàn toàn mới bằng cách sử dụng các câu chuyện khoa học viễn tưởng hiện có để đào tạo. Gần đây, Openai đã phát hành một mô hình ngôn ngữ được gọi là Openai-GPT-2, có thể tạo ra văn bản cực kỳ thực tế
- **Trả lời câu hỏi (Question Answering)**: Kỹ thuật QA có giá trị thương mại cao và các kỹ thuật như vậy được tìm thấy tại nền tảng của Chatbots và Virtual Assistant (ví dụ: Google Assistant và Apple Siri)
- **Dịch máy (Machine Translation)**: MT là nhiệm vụ chuyển đổi câu/cụm từ từ ngôn ngữ nguồn (ví dụ: tiếng Đức) sang ngôn ngữ đích (ví dụ: tiếng Anh). Đây là một nhiệm vụ rất khó khăn, vì các ngôn ngữ khác nhau có các cấu trúc cú pháp khác nhau, điều đó có nghĩa là nó không phải là một chuyển đổi một-một. Hơn nữa, các mối quan hệ từ ngữ giữa các ngôn ngữ có thể là một-nhiều, một-một, nhiều-một hoặc nhiều-nhiều.

Trong hình dưới, ta có thể thấy phân loại phân cấp của các nhiệm vụ NLP khác nhau được phân loại thành nhiều loại khác nhau. Đó là một nhiệm vụ khó khăn để gán một nhiệm vụ NLP cho một phân loại duy nhất. Chúng ta sẽ chia các danh mục thành hai loại chính: dựa trên ngôn ngữ (màu sáng với văn bản đen) và dựa trên công thức vấn đề (màu tối với văn bản trắng). Vấn đề ngôn ngữ có hai loại: cú pháp (dựa trên cấu trúc) và ngữ nghĩa (dựa trên ý nghĩa). Vấn đề dựa trên công thức có vấn đề có ba loại: các nhiệm vụ tiền xử lý (các tác vụ được thực hiện trên dữ liệu văn bản trước khi cung cấp cho một mô hình), nhiệm vụ phân loại(nhiệm vụ mà chúng ta cố gắng gán văn bản đầu vào cho một hoặc nhiều danh mục từ một tập hợp các danh mục được xác định trước) và các nhiệm vụ tạo ra (các nhiệm vụ mà chúng tôi cố gắng tạo ra một đầu ra văn bản mới)

![](/assets/img/NLP1.png)

## <font color = 'blue'> 3. Cách tiếp cận truyền thống để xử lý ngôn ngữ tự nhiên

Cách tiếp cận truyền thống hoặc cổ điển để giải quyết NLP là một luồng tuần tự của một số bước chính và đó là một cách tiếp cận thống kê. Khi chúng ta xem xét kỹ hơn về mô hình học tập NLP truyền thống, chúng ta sẽ có thể thấy một tập hợp các tác vụ riêng biệt đang diễn ra, chẳng hạn như tiền xử lý dữ liệu bằng cách xóa dữ liệu không mong muốn,kỹ thuật tính năng (feature engineering) để có được các biểu diễn tốt về dữ liệu văn bản, tìm hiểu để sử dụng các thuật toán học máy với sự trợ giúp của dữ liệu đào tạo và dự đoán đầu ra cho dữ liệu. Trong đó, kỹ thuật tính năng là bước tốn thời gian và quan trọng nhất để đạt được hiệu suất tốt trên một nhiệm vụ NLP nhất định.

## <font color = 'blue'> 3.Hiểu cách tiếp cận truyền thống

Đầu tiên,văn bản cần được xử lý trước, tập trung vào việc giảm từ vựng và phân tâm.Tiếp đến đến là một số bước kỹ thuật tính năng (feature engineering). Mục tiêu chính của kỹ thuật tính năng là làm cho việc học dễ dàng hơn cho các thuật toán. Thông thường các tính năng được thiết kế bằng tay và thiên vị đối với sự hiểu biết của con người về một ngôn ngữ. Kỹ thuật tính năng là vô cùng quan trọng đối với các thuật toán NLP cổ điển, và do đó, các hệ thống hiệu suất tốt nhất thường có các tính năng tốt nhất. Ví dụ: đối với một nhiệm vụ phân loại tình cảm, bạn có thể biểu thị một câu có cây phân tích và gán các nhãn dương, âm hoặc trung tính cho mỗi nút/cây con trong cây để phân loại câu đó là dương hoặc âm. Ngoài ra, giai đoạn kỹ thuật tính năng có thể sử dụng các tài nguyên bên ngoài như WordNet (cơ sở dữ liệu từ vựng có thể cung cấp hiểu biết về cách các từ khác nhau có liên quan với nhau - ví dụ: từ đồng nghĩa) để phát triển các tính năng tốt hơn. Chúng ta sẽ sớm xem xét một kỹ thuật kỹ thuật tính năng đơn giản được gọi là bag-of-words.

Tiếp theo, thuật toán học tập học cách thực hiện tốt trong nhiệm vụ đã cho bằng cách sử dụng các tính năng thu được và tùy chọn các tài nguyên bên ngoài. Ví dụ, đối với một nhiệm vụ tóm tắt văn bản, một kho văn bản song song chứa các cụm từ phổ biến và các chú giải cô đọng sẽ là một nguồn tài nguyên bên ngoài tốt. Cuối cùng, dự đoán xảy ra. Dự đoán rất đơn giản, trong đó bạn sẽ cung cấp một đầu vào mới và chứa các dự đoán bằng cách chuyển tiếp đầu vào thông qua mô hình học tập. Toàn bộ quá trình của phương pháp truyền thống được mô tả bên dưới
![](/assets/img/NLP2.png)

## <font color='blue'> 4.Hạn chế của phương pháp truyền thống
Các bước tiền xử lý được sử dụng trong NLP truyền thống buộc đánh đổi thông tin có khả năng hữu ích được nhúng trong văn bản (ví dụ: dấu câu) để làm cho việc học khả thi bằng cách giảm từ vựng.

Kỹ thuật tính năng mất rất nhiều thời gian. Để thiết kế một hệ thống đáng tin cậy, các tính năng tốt cần phải được nghĩ ra. Quá trình này có thể rất tẻ nhạt vì các không gian tính năng khác nhau cần được khám phá và đánh giá rộng rãi. Ngoài ra, để có hiệu quả các tính năng mạnh mẽ, cần có chuyên môn về miền, có thể khan hiếm và tốn kém cho một số nhiệm vụ NLP nhất định.

Các tài nguyên bên ngoài khác nhau là cần thiết để nó hoạt động tốt, và không có nhiều tài nguyên miễn phí. Các tài nguyên bên ngoài như vậy thường bao gồm thông tin được tạo thủ công được lưu trữ trong cơ sở dữ liệu lớn. Tạo một cho một nhiệm vụ cụ thể có thể mất vài năm, phụ thuộc vào mức độ nghiêm trọng của nhiệm vụ.

## <font color = 'blue'> 5.Phương pháp học sâu để xử lý ngôn ngữ tự nhiên

Các mô hình sâu đã tạo ra một làn sóng thay đổi mô hình trong nhiều lĩnh vực trong học máy, vì các mô hình sâu đã học được các tính năng phong phú từ dữ liệu thô thay vì sử dụng các tính năng bị hạn chế do con người thiết kế. Điều này do đó khiến kỹ thuật tính năng gây phiền nhiễu và đắt tiền bị lỗi thời. Với điều này, các mô hình sâu đã làm cho quy trình làm việc truyền thống hiệu quả hơn, vì các mô hình sâu thực hiện học tập tính năng và học tập nhiệm vụ một cách đồng thời. Hơn nữa, do số lượng lớn các tham số (nghĩa là trọng số) trong một mô hình sâu, nó có thể bao gồm nhiều tính năng hơn đáng kể so với con người có thể thiết kế. Tuy nhiên, các mô hình sâu được coi là một hộp đen do khả năng diễn giải kém của mô hình. 
Một mạng lưới thần kinh sâu về cơ bản là một mạng lưới thần kinh nhân tạo có lớp đầu vào, nhiều lớp ẩn được kết nối với nhau ở giữa, và cuối cùng, một lớp đầu ra (ví dụ: phân loại hoặc hồi quy). Như bạn có thể thấy, điều này tạo thành một mô hình từ đầu đến cuối từ dữ liệu thô đến dự đoán. Các lớp ẩn này ở giữa cung cấp sức mạnh cho các mô hình sâu vì chúng chịu trách nhiệm học các tính năng tốt từ dữ liệu thô.
## <font color = 'blue'> 6.Hiểu một mô hình sâu đơn giản - một mạng lưới thần kinh được kết nối đầy đủ
Bây giờ, hãy để có một cái nhìn kỹ hơn về một mạng lưới thần kinh sâu để có được sự hiểu biết tốt hơn. Mặc dù có rất nhiều biến thể khác nhau của các mô hình sâu, hãy nhìn vào một trong những mô hình sớm nhất có từ năm 1950 mô tả một a fully connected neural network (FCNN) có ba lớp tiêu chuẩn

Mục tiêu của FCNN là ánh xạ đầu vào (ví dụ: hình ảnh hoặc câu) đến một nhãn hoặc chú thích nhất định (ví dụ: danh mục đối tượng cho hình ảnh). Điều này đạt được bằng cách sử dụng đầu vào X để tính toán H - biểu diễn ẩn của X - sử dụng một phép biến đổi như ℎ=𝜎(𝑊 * h + b); Ở đây, W là trọng số và b là độ lệch của FCNN, và 𝜎 là chức năng kích hoạt sigmoid. Mạng thần kinh sử dụng các chức năng kích hoạt phi tuyến tính ở mỗi lớp. Kích hoạt sigmoid là một trong những kích hoạt như vậy. Nó là một chuyển đổi phần tử được áp dụng cho đầu ra của một lớp, trong đó đầu ra sigmoidal của x được cho bởi, 𝜎(𝑥) = 1/(1+𝑒-x). 

Tiếp theo, một trình phân loại được đặt trên đầu FCNN cung cấp khả năng tận dụng các tính năng đã học trong các lớp ẩn để phân loại đầu vào. Trình phân loại là một phần của FCNN và một lớp ẩn khác với một số trọng số Ws và thiên vị Bs. Ngoài ra, chúng ta có thể tính toán đầu ra cuối cùng của FCNN là output =softmax(𝑊𝑠 ∗ ℎ+ B𝑠).

Ví dụ, một phân loại SoftMax có thể được sử dụng cho các vấn đề phân loại đa nhãn. Nó cung cấp một biểu diễn chuẩn hóa của đầu ra điểm số của lớp phân loại. Đó là, nó sẽ tạo ra một phân phối xác suất hợp lệ trên các lớp trong lớp phân loại. Nhãn được coi là nút đầu ra có giá trị softmax cao nhất. Sau đó, với điều này, chúng ta có thể xác định tổn thất phân loại được tính là sự khác biệt giữa nhãn đầu ra dự đoán và nhãn đầu ra thực tế. Một ví dụ về chức năng mất mát như vậy là mất bình phương trung bình (mean squared loss).

Tiếp theo, các tham số mạng thần kinh, W, B, WS và BS, được tối ưu hóa bằng trình tối ưu hóa ngẫu nhiên tiêu chuẩn (ví dụ: giảm độ dốc ngẫu nhiên) để giảm sự mất phân loại của tất cả các đầu vào. Hình dưới mô tả quá trình được giải thích trong đoạn này cho FCNN ba lớp.

![](/assets/img/NLP3.png)

Có thể sử dụng mạng thần kinh (có thể sâu hoặc nông, tùy thuộc vào độ khó của nhiệm vụ) cho nhiệm vụ này bằng cách tuân thủ quy trình làm việc sau:

1. Mã thông báo (Tokenize) câu thành các từ. 
2. Chuyển đổi các câu thành một biểu diễn số có kích thước cố định (ví dụ: biểu diễn túi của các từ). Một biểu diễn có kích thước cố định là cần thiết vì các mạng thần kinh được kết nối đầy đủ đòi hỏi một đầu vào có kích thước cố định. 
3. Cung cấp các đầu vào số vào mạng thần kinh, dự đoán đầu ra (dương hoặc âm) và so sánh với mục tiêu thực. 
4. Tối ưu hóa mạng lưới thần kinh bằng cách sử dụng chức năng tổn thất mong muốn.

## <font color = 'blue'> 7. Một số nền tảng tính toán dựa trên đám mây phổ biến

• Google Colab: https://colab.research.google.com/

• Google Cloud Platform (GCP): https://cloud.google.com/

• Amazon Web Services (AWS): https://aws.amazon.com/

# <font color ='red'> Hiểu về Tensorflow 2

## <font color = 'blue'> Tensorflow là gì?

Tensorflow là một nguồn mở, được phát hành bởi Google, chủ yếu nhằm giảm bớt các chi tiết đau đớn khi thực hiện mạng lưới thần kinh (ví dụ, tính toán các dẫn xuất (derivatives) của trọng lượng của mạng lưới thần kinh). TensorFlow tiến thêm một bước bằng cách cung cấp các triển khai hiệu quả các tính toán số đó bằng cách sử dụng kiến trúc thiết bị hợp nhất tính toán (CUDA), là một nền tảng tính toán song song được NVIDIA giới thiệu. 

### <font color = 'blue'> Bắt đầu với TensorFlow 2

Bây giờ, hãy để tìm hiểu về một vài thành phần thiết yếu trong khung TensorFlow bằng cách làm việc thông qua một ví dụ về mã. Hãy để viết một ví dụ để thực hiện tính toán sau, điều này rất phổ biến đối với các mạng thần kinh:

![](/assets/img/NLP4.png)

Tính toán này bao gồm những gì xảy ra trong một lớp duy nhất của một mạng nơ-ron được kết nối hoàn toàn. Ở đây W và x là ma trận và b là một vectơ. Sau đó,".' biểu thị dot. sigmoid là một phép biến đổi phi tuyến tính được đưa ra bởi phương trình sau:

![](/assets/img/NLP5.png)

Đầu tiên, chúng ta sẽ cần nhập TensorFlow và Numpy. Numpy là một khung tính toán khoa học khác cung cấp các hoạt động toán học và các hoạt động khác để thao tác dữ liệu. Nhập vào chúng là điều cần thiết trước khi bạn chạy bất kỳ loại hoạt động liên quan đến TensorFlow hoặc Numpy trong Python:

```python
import tensorflow as tf 
import numpy as np
```

Đầu tiên, chúng tôi sẽ viết một hàm có thể lấy các đầu vào X, W và B và thực hiện tính toán này cho chúng tôi:

```python
def layer(x, W, b): 
    # Building the graph
    h = tf.nn.sigmoid(tf.matmul(x,W) + b) # Operation to perform
    return h
```

Tiếp theo, chúng tôi thêm một công cụ trang trí Python có tên TF.Function như sau

```python
@tf.function
def layer(x, W, b): 
    # Building the graph
    h = tf.nn.sigmoid(tf.matmul(x,W) + b) # Operation to perform
    return h
```

Nói một cách đơn giản, một decorator Python chỉ là một hàm khác. Một decorator Python cung cấp một cách sạch sẽ để gọi một hàm khác bất cứ khi nào bạn gọi hàm decorated. Nói cách khác, mỗi khi hàm layer() được gọi, tf.function() được gọi. Điều này có thể được sử dụng cho các mục đích khác nhau, chẳng hạn như:

• Ghi nhật ký nội dung và hoạt động trong một hàm 

• Xác thực các đầu vào và đầu ra của hàm khác

Khi hàm layer() đi qua tf.function (), TensorFlow sẽ theo dõi nội dung (nói cách khác, các hoạt động và dữ liệu) trong hàm và tự động xây dựng biểu đồ tính toán.

Biểu đồ tính toán (còn được gọi là biểu đồ DataFlow) xây dựng DAG (biểu đồ acyclic có hướng) hiển thị loại đầu vào nào được yêu cầu và loại tính toán cần được thực hiện trong chương trình

Trong ví dụ này, hàm layer() tạo ra H bằng cách sử dụng đầu vào X, W và B và một số phương tiện chuyển đổi hoặc các operations như + và tf.matmul ():

![](/assets/img/NLP6.png)

Nếu chúng ta nhìn vào một sự tương tự cho một DAG, nếu bạn nghĩ về đầu ra như 
một chiếc bánh, thì biểu đồ sẽ là công thức để làm cho bánh đó sử dụng các thành phần (nghĩa là đầu vào).

Tính năng xây dựng biểu đồ tính toán này tự động trong TensorFlow được gọi là AutoGraph. AutoGraph không chỉ nhìn vào các operations trong hàm được thông qua; Nó cũng xem xét kỹ lưỡng dòng chảy của các hoạt động. Điều này có nghĩa là bạn có thể có nếu các câu lệnh if, hoặc các vòng lặp for/while trong hàm của bạn và AutoGraph sẽ quan tâm chúng khi xây dựng biểu đồ. 

Tiếp theo, bạn có thể sử dụng chức năng này ngay lập tức, như sau

```python
x = np.array([[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],dtype=np.float32)
```

Ở đây, x là một mảng numpy đơn giản

```python
init_w = tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)(shape=[10,5])
W = tf.Variable(init_w, dtype=tf.float32, name='W') 
init_b = tf.initializers.RandomUniform()(shape=[5])
b = tf.Variable(init_b, dtype=tf.float32, name='b')
```

W và B là các biến TensorFlow được xác định bằng đối tượng TF.Varable. W và B là tensor. Một tensor về cơ bản là một mảng N chiều. Ví dụ, một vectơ một chiều hoặc ma trận hai chiều được gọi là tensor. tf.variable là một cấu trúc có thể thay đổi, có nghĩa là các giá trị trong tensor được lưu trữ trong biến đó có thể thay đổi theo thời gian. Ví dụ, các biến được sử dụng để lưu trữ các trọng số mạng thần kinh, thay đổi trong quá trình tối ưu hóa mô hình.

Ngoài ra, lưu ý rằng với W và B, chúng tôi cung cấp một số đối số quan trọng, chẳng hạn như sau:

```python
init_w = tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)(shape=[10,5])
init_b = tf.initializers.RandomUniform()(shape=[5])
```

Chúng được gọi là các bộ khởi tạo biến và là các tensor sẽ được gán cho các biến W và B ban đầu. Một biến phải có một giá trị ban đầu được cung cấp. Ở đây, tf.initializer.randomuniform có nghĩa là chúng ta thống nhất các giá trị mẫu giữa minval (-0.1) và maxval (0,1) để gán các giá trị cho các tensor. Có nhiều bộ khởi tạo khác nhau được cung cấp trong TensorFlow (https: // www.tensorflow.org/api_docs/python/tf/keras/initializer). Nó cũng rất quan trọng để xác định hình dạng của bộ khởi tạo của bạn khi bạn đang xác định chính bộ khởi tạo. Thuộc tính hình dạng xác định kích thước của từng chiều của tenxơ đầu ra. Ví dụ: nếu hình dạng là [10, 5], điều này có nghĩa là nó sẽ là cấu trúc hai chiều và sẽ có 10 phần tử trên trục 0 (hàng) và 5 phần tử trên trục 1 (cột):

```python
h = layer(x,W,b)
```

Cuối cùng, H được gọi là tensorflow tensor nói chung. Một tensorflow tensor là một cấu trúc bất biến. Khi một giá trị được gán cho tensorflow tensor, nó không thể thay đổi.

Như bạn có thể thấy, thuật ngữ tensor được sử dụng theo hai cách:

 • để chỉ một mảng n chiều 
 
 • để tham khảo cấu trúc dữ liệu bất biến trong tensorflow

Cuối cùng, bạn có thể thấy ngay giá trị của H

```python
print(f"h = {h.numpy()}")
```

```python
@tf.function
def layer(x, W, b): 
    # Building the graph
    h = tf.nn.sigmoid(tf.matmul(x,W) + b) # Operation to be performed
    return h
x = np.array([[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]], 
dtype=np.float32) 
# Variable
init_w = tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)(shape=[10,5])
W = tf.Variable(init_w, dtype=tf.float32, name='W') 
# Variable
init_b = tf.initializers.RandomUniform()(shape=[5])
b = tf.Variable(init_b, dtype=tf.float32, name='b') 
h = layer(x,W,b)
print(f"h = {h.numpy()}")
```


