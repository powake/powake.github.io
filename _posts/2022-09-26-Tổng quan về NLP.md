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

# <font color ='red'> II.Hiểu về Tensorflow 2

## <font color = 'blue'> 1.Tensorflow là gì?

Tensorflow là một nguồn mở, được phát hành bởi Google, chủ yếu nhằm giảm bớt các chi tiết đau đớn khi thực hiện mạng lưới thần kinh (ví dụ, tính toán các dẫn xuất (derivatives) của trọng lượng của mạng lưới thần kinh). TensorFlow tiến thêm một bước bằng cách cung cấp các triển khai hiệu quả các tính toán số đó bằng cách sử dụng kiến trúc thiết bị hợp nhất tính toán (CUDA), là một nền tảng tính toán song song được NVIDIA giới thiệu. 

## <font color = 'blue'> 2.Bắt đầu với TensorFlow 2

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
## <font color = 'blue'> 3.Đầu vào, biến, đầu ra và operation

• Đầu vào: Dữ liệu được sử dụng để đào tạo và kiểm tra các thuật toán của chúng tôi 

• Biến: Tensor có thể thay đổi, chủ yếu xác định các tham số của thuật toán của chúng tôi

• Đầu ra: Các tensor bất biến lưu trữ cả đầu ra đầu cuối và đầu ra trung gian 

• Operation: Các phép biến đổi khác nhau cho đầu vào để tạo ra đầu ra mong muốn

![](/assets/img/NLP7.png)

### <font color = 'green'> Định nghĩa đầu vào trong Tensorflow
Có ba cách khác nhau mà bạn có thể cung cấp dữ liệu cho chương trình TensorFlow: 

• Tạo dữ liệu dưới dạng mảng Numpy 

• Tạo dữ liệu dưới dạng tenorflow tensors 

• Sử dụng API TF.DATA để tạo đường ống đầu vào 

### <font color = 'green'> Định nghĩa biến trong Tensorflow
Các biến đóng một vai trò quan trọng trong tensorflow. Một biến về cơ bản là một tenxơ với hình dạng cụ thể xác định có bao nhiêu kích thước mà biến sẽ có và kích thước của mỗi chiều. Tuy nhiên, không giống như một tenxơ tenorflow thông thường, các biến có thể thay đổi; nghĩa là giá trị của các biến có thể thay đổi sau khi chúng được xác định. Đây là một thuộc tính lý tưởng để phải thực hiện các tham số của mô hình học tập (ví dụ: trọng lượng mạng thần kinh), trong đó các trọng số thay đổi một chút sau mỗi bước học tập. Ví dụ: nếu bạn xác định một biến có x = tf.varable (0, dtype = tf.int32), bạn có thể thay đổi giá trị của biến đó bằng cách sử dụng hoạt động tenorflow như tf.assign (x, x+1). Tuy nhiên, nếu bạn xác định một tenxơ như x = tf.constant (0, dtype = tf.int32), bạn không thể thay đổi giá trị của tenxơ, như bạn có thể cho một biến. Nó sẽ ở lại 0 cho đến khi kết thúc việc thực hiện chương trình.

Tạo biến là khá đơn giản. Trong ví dụ sigmoid của chúng ta, chúng ta đã tạo hai biến, W và b. Khi tạo ra một biến, một vài điều là vô cùng quan trọng. Chúng ta sẽ liệt kê chúng ở đây và thảo luận chi tiết trong các đoạn sau:

- Hình dạng biến 
- Giá trị ban đầu 
- Kiểu dữ liệu 
- Tên (Tùy chọn)

Hình dạng biến là một danh sách của định dạng [x, y, z, ...]. Mỗi giá trị trong danh sách cho biết kích thước hoặc trục tương ứng lớn như thế nào. Chẳng hạn, nếu bạn yêu cầu tenxơ 2D với 50 hàng và 10 cột làm biến, hình dạng sẽ bằng [50,10].

Kích thước của biến (nghĩa là độ dài của vectơ hình dạng) được công nhận là thứ hạng của tensor trong tenorflow.

Tiếp theo, một biến yêu cầu một giá trị ban đầu phải được khởi tạo. TensorFlow cung cấp một số bộ khởi tạo khác nhau, bao gồm các bộ khởi tạo không đổi và bộ khởi tạo phân phối bình thường. Dưới đây là một vài bộ khởi tạo TensorFlow phổ biến mà bạn có thể sử dụng để khởi tạo các biến:

- tf.initializers.Zeros
- tf.initializers.Constant
- tf.initializers.RandomNormal
- tf.initializers.GlorotUniform

Hình dạng của biến có thể được cung cấp như là một phần của trình khởi tạo như sau:

```python
tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)(shape=[10,5])
```

Kiểu dữ liệu đóng một vai trò quan trọng trong việc xác định kích thước của một biến. Có nhiều loại dữ liệu khác nhau, bao gồm TF.Bool, TF.Uint8, TF.Float32 và TF.INT32. Mỗi loại dữ liệu có một số bit cần thiết để biểu diễn một giá trị duy nhất với loại đó. Ví dụ, tf.uint8 yêu cầu 8 bit, trong khi tf.float32 yêu cầu 32 bit. Đó là thực tế phổ biến để sử dụng các loại dữ liệu tương tự cho các tính toán, vì làm cách khác có thể dẫn đến sự không phù hợp của kiểu dữ liệu. Vì vậy, nếu bạn có hai loại dữ liệu khác nhau cho hai tenxor mà bạn cần chuyển đổi, bạn phải chuyển đổi rõ ràng một tenxơ sang loại tenxơ khác bằng cách sử dụng thao tác tf.cast (...).

Hoạt động tf.cast (...) được thiết kế để đối phó với các tình huống như vậy. Ví dụ: nếu bạn có biến X với loại tf.int32, cần được chuyển đổi thành tf.float32, sử dụng tf.cast(x, dtype = tf.float32) để chuyển đổi x thành tf.float32.

Cuối cùng, tên của biến sẽ được sử dụng làm ID để xác định biến đó trong biểu đồ. Nếu bạn đã từng trực quan hóa biểu đồ tính toán, biến sẽ xuất hiện bằng đối số được chuyển đến tên từ khóa. Nếu bạn không chỉ định tên, TensorFlow sẽ sử dụng sơ đồ đặt tên mặc định.

```python
a = tf.Variable(tf.zeros([5]),name='b')
```

Ở đây, biểu đồ tensorflow sẽ biết biến này bằng tên b chứ không phải a

### <font color = 'green'> Định nghĩa đầu ra trong Tensorflow

Đầu ra tenorflow thường là tensor và kết quả của việc chuyển đổi thành đầu vào hoặc một biến hoặc cả hai. Trong ví dụ của chúng ta, h là đầu ra, trong đó h = tf.nn.sigmoid (tf.matmul (x, w) + b). Cũng có thể cung cấp các đầu ra như vậy cho các hoạt động khác, tạo thành một tập hợp các hoạt động chuỗi. Hơn nữa, không nhất thiết phải là hoạt động tensorflow. Bạn cũng có thể sử dụng số học python tiêu chuẩn với tensorflow. Đây là một ví dụ:

```python
x = tf.matmul(w,A)
y = x + B
```

### <font color = 'green'> Định nghĩa operations trong TensorFlow

Một hoạt động trong TensorFlow có một hoặc nhiều đầu vào và tạo ra một hoặc nhiều đầu ra. Nếu bạn xem API TensorFlow tại https://www.tensorflow.org/api_docs/python/tf, bạn sẽ thấy TensorFlow có một bộ sưu tập hoạt động lớn. Ở đây, chúng tôi sẽ xem xét một vài trong số các hoạt động vô số tenorflow.

#### <font color = 'pink'> Hoạt động so sánh

Hoạt động so sánh rất hữu ích để so sánh hai tensor. Ví dụ mã sau đây bao gồm một vài hoạt động so sánh hữu ích.

```python
import tensorflow as tf
x = tf.constant([[1,2],[3,4]], dtype=tf.int32)
y = tf.constant([[4,3],[3,2]], dtype=tf.int32)

x_equal_y = tf.equal(x, y, name=None)

x_less_y = tf.less(x, y, name=None)

x_great_equal_y = tf.greater_equal(x, y, name=None)

condition = tf.constant([[True,False],[True,False]],dtype=tf.bool)

x_cond_y = tf.where(condition, x, y, name=None)
```

#### <font color='pink'> Hoạt động toán học

TensorFlow cho phép bạn thực hiện các thao tác toán học trên các tenxơ từ đơn giản đến phức tạp. Bộ hoạt động hoàn chỉnh có sẵn tại https://www.tensorflow.org/versions/r2.0/ api_docs/python/tf/math:

```python
x = tf.constant([[1,2],[3,4]], dtype=tf.float32)
y = tf.constant([[4,3],[3,2]], dtype=tf.float32)

x_add_y = tf.add(x, y)

x_mul_y = tf.matmul(x, y)

log_x = tf.log(x)

x_sum_1 = tf.reduce_sum(x, axis=[1], keepdims=False)

x_sum_2 = tf.reduce_sum(x, axis=[0], keepdims=True)

data = tf.constant([1,2,3,4,5,6,7,8,9,10], dtype=tf.float32)
segment_ids = tf.constant([0,0,0,1,1,2,2,2,2,2 ], dtype=tf.int32)

x_seg_sum = tf.segment_sum(data, segment_ids)
```

#### <font color = 'pink'> Cập nhật giá trị trong các tensor

Một hoạt động phân tán (scatter operation), đề cập đến việc thay đổi các giá trị tại một số chỉ số nhất định của một tensor, là rất phổ biến trong các vấn đề điện toán khoa học. Chức năng này ban đầu được cung cấp thông qua hàm tf.scatter_nd() 

Tuy nhiên, trong các phiên bản TensorFlow gần đây, bạn có thể thực hiện các hoạt động phân tán thông qua lập chỉ mục mảng và cắt bằng cú pháp giống như Numpy. Hãy cùng xem một vài ví dụ. Giả sử bạn có TensorFlow biến V, là ma trận [3,2]:

```python
v = tf.Variable(tf.constant([[1,9],[3,10],[5,11]],
dtype=tf.float32),name='ref')   
```

Bạn có thể thay đổi hàng thứ 0 của tenxơ này bằng:

```python
v[0].assign([-1, -9])
```

Bạn có thể thay đổi giá trị tại Index [1,1] bằng:

```python
v[1,1].assign(-10)
```

Bạn có thể thực hiện cắt hàng với:

```python
v[1:,0].assign([-3,-5])
```

#### <font color = 'pink'> Thu thập các giá trị từ một tenor

Một hoạt động tập hợp (gather operation) rất giống với một hoạt động phân tán. Hãy nhớ rằng phân tán là về việc gán các giá trị cho các tensor, trong khi việc thu thập lấy các giá trị của một tensor. Hãy để hiểu điều này thông qua một ví dụ. Giả sử bạn có tenorflow tenor, T:

```python
t = tf.constant([[1,9],[3,10],[5,11]],dtype=tf.float32)
```

Bạn có thể có được hàng thứ 0 của T với:

```python
t[0].numpy()
```

Bạn cũng có thể thực hiện trượt hàng (row-slicing) với:

```python
t[1:,0].numpy()
```

Không giống như hoạt động phân tán, hoạt động tập hợp hoạt động cả trên các cấu trúc TF.Varable và TF.Tensor.

## <font color = 'blue'> 4.Operation liên quan đến mạng thần kinh

Bây giờ, hãy xem xét một số hoạt động liên quan đến mạng thần kinh hữu ích mà chúng ta sẽ sử dụng rất nhiều trong các chương sau. Các hoạt động mà chúng tôi sẽ thảo luận ở đây bao gồm từ các biến đổi phần tử đơn giản (nghĩa là kích hoạt) đến tính toán các dẫn xuất một phần của một tập hợp các tham số đối với giá trị khác. Chúng ta cũng sẽ triển khai một mạng lưới thần kinh đơn giản.

### <font color = 'green'> Kích hoạt phi tuyến được sử dụng bởi các mạng thần kinh

Kích hoạt phi tuyến cho phép các mạng thần kinh hoạt động tốt ở nhiều nhiệm vụ. Thông thường, có một phép biến đổi kích hoạt phi tuyến (nghĩa là lớp kích hoạt) sau mỗi đầu ra lớp trong mạng thần kinh (ngoại trừ lớp cuối cùng). Một phép biến đổi phi tuyến giúp một mạng lưới thần kinh tìm hiểu các mẫu phi tuyến khác nhau có trong dữ liệu. Điều này rất hữu ích cho các vấn đề trong thế giới thực phức tạp, trong đó dữ liệu thường có các mẫu phi tuyến phức tạp hơn, trái ngược với các mẫu tuyến tính. Nếu không dành cho các kích hoạt phi tuyến giữa các lớp, một mạng lưới thần kinh sâu sẽ là một loạt các lớp tuyến tính được xếp chồng lên nhau. Ngoài ra, một tập hợp các lớp tuyến tính về cơ bản có thể được nén vào một lớp tuyến tính lớn hơn.

Tóm lại, nếu không cho các kích hoạt phi tuyến, chúng ta không thể tạo ra một mạng lưới thần kinh với nhiều hơn một lớp.

Tầm quan trọng của việc kích hoạt phi tuyến thông qua một ví dụ. Đầu tiên, hãy nhớ lại việc tính toán cho các mạng thần kinh mà chúng ta đã thấy trong ví dụ SigMoid.

```python
h = sigmoid(W*x)
```

Giả sử một mạng lưới thần kinh ba lớp (có W1, W2 và W3 làm trọng số lớp) trong đó mỗi lớp thực hiện tính toán trước đó; Chúng ta có thể tóm tắt tính toán đầy đủ như sau

```python
h = sigmoid(W3*sigmoid(W2*sigmoid(W1*x)))
```

Tuy nhiên, nếu chúng ta loại bỏ kích hoạt phi tuyến (nghĩa là sigmoid), chúng ta sẽ nhận được điều này:

```python
h = (W3 * (W2 * (W1 *x))) = (W3*W2*W1)*x
```

Vì vậy, không có kích hoạt phi tuyến, ba lớp có thể được đưa xuống một lớp tuyến tính duy nhất

Bây giờ chúng tôi sẽ liệt kê hai kích hoạt phi tuyến ( nonlinear activations) thường được sử dụng trong các mạng thần kinh (nói cách khác là SigMoid và Relu) và cách chúng có thể được thực hiện trong TensorFlow

```python
# Sigmoid : 1 / (1 + exp(-x))
tf.nn.sigmoid(x,name=None)
# ReLU activation : max(0,x)
tf.nn.relu(x, name=None)
```

![](/assets/img/NLP8.png)
  
### <font color = 'green'> Convolution operation

Một hoạt động tích chập là một kỹ thuật xử lý tín hiệu được sử dụng rộng rãi. Đối với hình ảnh, tích chập được sử dụng để tạo ra các hiệu ứng khác nhau (như làm mờ) hoặc trích xuất các tính năng (như các cạnh) từ một hình ảnh. Một ví dụ về phát hiện cạnh bằng cách sử dụng tích chập được hiển thị trong Hình dưới. Điều này đạt được bằng cách chuyển một bộ lọc tích chập của hình ảnh để tạo ra một đầu ra khác nhau ở mỗi vị trí. Cụ thể, tại mỗi vị trí, chúng tôi thực hiện phép nhân phần tử của các phần tử trong bộ lọc tích chập với bản vá hình ảnh (image patch) (cùng kích thước với bộ lọc tích chập) trùng với bộ lọc tích chập và lấy tổng của phép nhân
  
![](/assets/img/NLP9.png)
  
Sau đây là việc thực hiện hoạt động tích chập

```python
x = tf.constant(
 [[
 [[1],[2],[3],[4]],
 [[4],[3],[2],[1]],
 [[5],[6],[7],[8]],
 [[8],[7],[6],[5]]
 ]],
 dtype=tf.float32)
x_filter = tf.constant(
 [ [ [[0.5]],[[1]] ],
 [ [[0.5]],[[1]] ]
 ],
 dtype=tf.float32)
x_stride = [1,1,1,1]
x_padding = 'VALID'
x_conv = tf.nn.conv2d(
 input=x, filters=x_filter, strides=x_stride, padding=x_padding
)
```
  
Đối với hoạt động tf.nn.conv2d (...), TensorFlow yêu cầu đầu vào, bộ lọc và sải bước ( input, filters, and strides ) có định dạng chính xác. Bây giờ chúng ta sẽ đi qua từng đối số trong tf.conv2d (đầu vào, bộ lọc, sải chân, đệm) ((input, filters, strides, padding)) chi tiết hơn:

Input : Đây thường là một tenxơ 4D trong đó các kích thước nên được đặt dưới dạng [batch_size, height, width, channels]:
- Batch_Size: Đây là lượng dữ liệu (ví dụ: các đầu vào như hình ảnh và từ) trong một lô dữ liệu. Chúng ta thường xử lý dữ liệu theo lô vì các bộ dữ liệu lớn được sử dụng để học. Ở một bước đào tạo nhất định, chúng ta lấy mẫu ngẫu nhiên một lô dữ liệu nhỏ đại diện cho bộ dữ liệu đầy đủ. Và làm điều này cho nhiều bước cho phép chúng ta xấp xỉ bộ dữ liệu đầy đủ khá tốt. Tham số Batch_Size này giống như tham số chúng ta đã thảo luận trong ví dụ đường ống đầu vào TensorFlow.
- Height and width: Đây là chiều cao và chiều rộng của đầu vào
- Chanels: Đây là độ sâu của đầu vào (ví dụ: đối với hình ảnh RGB, số lượng kênh sẽ là 3 kênh, một kênh cho mỗi màu).

Bộ lọc: Đây là một tenxơ 4D đại diện cho cửa sổ tích chập của hoạt động tích chập. Kích thước bộ lọc phải là [height, width, in_channels, out_channels]:
- Height and width: Đây là chiều cao và chiều rộng của bộ lọc (thường nhỏ hơn so với đầu vào)
- in_channels: Đây là số lượng kênh đầu vào cho lớp
- out_channels: Đây là số lượng kênh được sản xuất trong đầu ra của lớp

strides: Đây là danh sách với bốn yếu tố, trong đó các phần tử là [batch_stride, height_stride, width_stride, channels_stride]. Đối số Strides biểu thị có bao nhiêu phần tử cần bỏ qua trong một dịch chuyển của cửa sổ tích chập trên đầu vào. Thông thường, bạn không phải lo lắng về Batch_Stride và channels_stride. Nếu bạn không hoàn toàn hiểu strides (bước tiến) là gì, bạn có thể sử dụng giá trị mặc định là 1.

Padding: Đây có thể là một trong số ['SAME', 'VALID']. Nó quyết định làm thế nào để xử lý hoạt động tích chập gần ranh giới của đầu vào. Các hoạt động hợp lệ (VALID) thực hiện tích chập mà không cần đệm (padding). Nếu chúng ta kết hợp một đầu vào có độ dài n với một cửa sổ tích chập có kích thước H, điều này sẽ dẫn đến đầu ra có kích thước (N-H+1 <N). Việc giảm kích thước đầu ra có thể hạn chế nghiêm trọng độ sâu của mạng lưới thần kinh. SAME thêm các số 0 đến ranh giới sao cho đầu ra sẽ có cùng chiều cao và chiều rộng với đầu vào.

Để hiểu rõ hơn về kích thước bộ lọc, sải chân và đệm (filter size, stride, and padding), tham khảo hình dưới

![](/assets/img/NLP10.png)

![](/assets/img/NLP11.png)

![](/assets/img/NLP12.png)

### <font color = 'green'> Pooling operation

Một hoạt động gộp (pooling operation) hoạt động tương tự như hoạt động tích chập, nhưng đầu ra cuối cùng là khác nhau. Thay vì xuất tổng số nhân của bộ lọc và bản vá hình ảnh, giờ đây chúng ta lấy phần tử tối đa của bản vá hình ảnh cho vị trí đó.

```python
x = tf.constant(
 [[
 [[1],[2],[3],[4]],
 [[4],[3],[2],[1]],
 [[5],[6],[7],[8]],
 [[8],[7],[6],[5]]
 ]],
 dtype=tf.float32)
x_ksize = [1,2,2,1]
x_stride = [1,2,2,1]
x_padding = 'VALID'
x_pool = tf.nn.max_pool2d(
 input=x, ksize=x_ksize,
 strides=x_stride, padding=x_padding
)
# Returns (out) => [[[[ 4.],[ 4.]],[[ 8.],[ 8.]]]]
```
![](/assets/img/NLP13.png)
  
### <font color = 'green'> Định nghĩa mất mát

Chúng ta biết rằng, đối với một mạng lưới thần kinh để học một cái gì đó hữu ích, một mất mát cần phải được xác định. Sự mất mát thể hiện mức độ gần hoặc xa các dự đoán từ các mục tiêu thực tế. Có một số chức năng để tự động tính toán tổn thất trong tensorflow, hai trong số đó được hiển thị trong mã sau. Hàm tf.nn.l2_loss là mất lỗi bình phương trung bình (mean squared error loss) và tf.nn.softmax_cross_entropy_with_logits là một loại tổn thất khác thực sự mang lại hiệu suất tốt hơn trong các tác vụ phân loại. 

```python
# Returns half of L2 norm of t given by sum(t**2)/2
x = tf.constant([[2,4],[6,8]],dtype=tf.float32)
x_hat = tf.constant([[1,2],[3,4]],dtype=tf.float32)
# MSE = (1**2 + 2**2 + 3**2 + 4**2)/2 = 15
MSE = tf.nn.l2_loss(x-x_hat)

y = tf.constant([[1,0],[0,1]],dtype=tf.float32)
y_hat = tf.constant([[3,1],[2,5]],dtype=tf.float32)

CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat,labels=y))
```

## <font color = 'blue'> 5.Keras: API xây dựng mô hình của Tensorflow

Keras được phát triển như một thư viện riêng biệt cung cấp các khối xây dựng cấp cao để xây dựng các mô hình một cách thuận tiện. Ban đầu nó hỗ trợ nhiều phần mềm (ví dụ: Tensorflow và Theano). Tuy nhiên, Tensorflow có được Keras và bây giờ là một phần không thể thiếu trong TensorFlow để xây dựng các mô hình một cách dễ dàng.

Trọng tâm chính của Keras là xây dựng mô hình. Vì vậy, Keras cung cấp một số API khác nhau với mức độ linh hoạt và phức tạp khác nhau. Chọn API phù hợp cho công việc sẽ yêu cầu kiến thức hợp lý về các hạn chế của mỗi API cũng như kinh nghiệm. Các API được cung cấp bởi Keras là:

- API tuần tự (Sequential API) : API dễ sử dụng nhất. Trong API này, bạn chỉ cần xếp các lớp lên nhau để tạo một mô hình. 
- API chức năng (Functional API) - API chức năng cung cấp tính linh hoạt hơn bằng cách cho phép bạn xác định các mô hình tùy chỉnh có thể có nhiều lớp đầu vào/nhiều lớp đầu ra. 
- API lớp phụ (Sub-classing API) : API lớp phụ cho phép bạn xác định các lớp/ mô hình có thể tái sử dụng tùy chỉnh là các lớp Python. Đây là API linh hoạt nhất, nhưng nó đòi hỏi sự quen thuộc mạnh mẽ với các hoạt động API và tensorflow thô để sử dụng nó một cách chính xác

Một trong những khái niệm bẩm sinh nhất trong Keras là một mô hình bao gồm một hoặc nhiều lớp được kết nối theo một cách cụ thể. Ở đây, chúng ta sẽ ngắn gọn về mã trông như thế nào, sử dụng các API khác nhau để phát triển các mô hình. Bạn không mong đợi hiểu đầy đủ mã dưới đây. Thay vào đó, tập trung vào kiểu mã để phát hiện ra bất kỳ sự khác biệt nào giữa ba phương pháp
  
### <font color = 'green'> Sequential API

Khi sử dụng API tuần tự, bạn chỉ cần xác định mô hình của mình là danh sách các lớp. Ở đây, phần tử đầu tiên trong danh sách là gần nhất với đầu vào, trong đó phần cuối là lớp đầu ra:

```python
model = tf.keras.Sequential([
 tf.keras.layers.Dense(500, activation='relu', shape=(784, )),
 tf.keras.layers.Dense(250, activation='relu'),
 tf.keras.layers.Dense(10, activation='softmax')
 ])
```

Trong mã trước, chúng tôi có ba lớp. Lớp đầu tiên có 500 nút đầu ra và lấy một vectơ gồm 784 phần tử làm đầu vào. Lớp thứ hai được tự động kết nối với lớp thứ nhất, trong khi lớp cuối cùng được kết nối với lớp thứ hai. Tất cả các lớp này là các lớp được kết nối đầy đủ, trong đó tất cả các nút đầu vào được kết nối với tất cả các nút đầu ra.

### <font color = 'green'> Functional API

Trong API chức năng, chúng ta làm mọi thứ khác nhau. Trước tiên chúng ta xác định một hoặc nhiều lớp đầu vào và các lớp khác mang tính toán. Sau đó, chúng tôi kết nối các đầu vào với đầu ra, như được hiển thị trong mã sau:

```python
inp = tf.keras.layers.Input(shape=(784,))
out_1 = tf.keras.layers.Dense(500, activation='relu')(inp)
out_2 = tf.keras.layers.Dense(250, activation='relu')(out_1)
out = tf.keras.layers.Dense(10, activation='softmax')(out_2)
model = tf.keras.models.Model(inputs=inp, outputs=out)
```

Trong mã, chúng ta bắt đầu với một lớp đầu vào chấp nhận vectơ dài 784 phần tử. Đầu vào được truyền đến một lớp dày đặc có 500 nút. Đầu ra của lớp đó được gán cho out_1. Sau đó out_1 được chuyển cho một lớp dày đặc khác, xuất ra out_2. Tiếp theo, một lớp dày đặc với 10 nút đầu ra đầu ra cuối cùng. Cuối cùng, mô hình được định nghĩa là đối tượng tf.keras.models.Model có hai đối số:

- inputs - một hoặc nhiều lớp đầu vào  
- outputs - một hoặc nhiều đầu ra được tạo bởi bất kỳ tf.keras.layers loại đối tượng

Mô hình giống hệt với những gì được xác định trong phần trước. Một trong những lợi ích của API chức năng là bạn có thể tạo các mô hình phức tạp hơn nhiều vì bạn không bị ràng buộc để có các lớp như một danh sách. Vì sự tự do này, bạn có thể có nhiều đầu vào kết nối với nhiều lớp theo nhiều cách khác nhau và có khả năng tạo ra nhiều đầu ra.



### <font color = 'green'> Sub-classing API

Cuối cùng, chúng ta sẽ sử dụng API lớp phụ để xác định mô hình. Với lớp phụ, bạn xác định mô hình của mình là một đối tượng Python kế thừa từ đối tượng cơ sở tf.keras.model. Khi sử dụng lớp phụ, bạn cần xác định hai hàm quan trọng: __init __ (), sẽ chỉ định bất kỳ tham số, lớp đặc biệt nào, và do đó cần thiết để thực hiện thành công các tính toán và hàm  call() xác định các tính toán cần phải xảy ra trong mô hình:

```python
class MyModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.hidden1_layer = tf.keras.layers.Dense(500, activation='relu')
        self.hidden2_layer = tf.keras.layers.Dense(250, activation='relu')
        self.final_layer = tf.keras.layers.Dense(num_classes,
        activation='softmax')
    def call(self, inputs):
        h = self.hidden1_layer(inputs)
        h = self.hidden2_layer(h)
        y = self.final_layer(h)
        return y


model = MyModel(num_classes=10)
```
  
Ở đây, bạn có thể thấy rằng mô hình của chúng ta có ba lớp, giống như tất cả các mô hình trước đó chúng ta đã xác định. Tiếp theo, hàm call() xác định cách các lớp này kết nối để tạo ra đầu ra cuối cùng. API lớp phụ được coi là khó khăn nhất để làm chủ, chủ yếu là do sự tự do. Tuy nhiên, phần thưởng là rất lớn khi bạn tìm hiểu API vì nó cho phép bạn xác định các mô hình/lớp rất phức tạp là các tính toán đơn vị có thể được sử dụng lại sau đó. Bây giờ bạn đã hiểu cách mỗi API hoạt động, hãy để thực hiện một mạng lưới thần kinh bằng cách sử dụng Keras và đào tạo nó trên một bộ dữ liệu.
 
## <font color = 'blue'> 6.Thực hiện mạng neural network đầu tiên của chúng ta

Một trong những bước đệm để giới thiệu các mạng thần kinh là triển khai một mạng lưới thần kinh có khả năng phân loại các chữ số. Đối với nhiệm vụ này, chúng tôi sẽ sử dụng bộ dữ liệu MNIST nổi tiếng được cung cấp tại http://yann.lecun.com/exdb/mnist/.

Bạn có thể cảm thấy một chút hoài nghi về việc chúng ta sử dụng nhiệm vụ tầm nhìn máy tính hơn là một nhiệm vụ NLP. Tuy nhiên, các nhiệm vụ tầm nhìn có thể được thực hiện với ít tiền xử lý hơn và dễ hiểu.

Vì đây là cuộc gặp gỡ đầu tiên của chúng ta với các mạng thần kinh, chúng ta sẽ thấy cách thực hiện mô hình này bằng cách sử dụng Keras. Keras là mô hình con cấp cao cung cấp một lớp trừu tượng qua tensorflow. Do đó, bạn có thể triển khai các mạng thần kinh với ít nỗ lực hơn với Keras hơn là sử dụng các hoạt động thô của TensorFlow. 
  
### <font color = 'green'> Chuẩn bị dữ liệu

Đầu tiên, chúng ta cần tải xuống bộ dữ liệu. TensorFlow cung cấp các chức năng thuận tiện để tải xuống dữ liệu và MNIST là một trong những bộ dữ liệu được hỗ trợ đó. Chúng tôi sẽ thực hiện bốn bước quan trọng trong quá trình chuẩn bị dữ liệu:

- Tải xuống dữ liệu và lưu trữ nó dưới dạng các đối tượng numpy.ndarray. 
- Định hình lại các hình ảnh để hình ảnh thang độ xám 2D trong bộ dữ liệu sẽ được chuyển đổi thành vectơ 1D. 
- Tiêu chuẩn hóa các hình ảnh có trung bình không và đơn vị (zero-mean and unit-variance) (còn được gọi là làm trắng). 
- One-hot encoding nhãn lớp số nguyên. Mã hóa một lần đề cập đến quá trình biểu diễn nhãn lớp số nguyên dưới dạng vectơ. Ví dụ: nếu bạn có 10 lớp và nhãn lớp 3 (trong đó các nhãn nằm trong khoảng từ 0-9), vectơ được mã hóa một lần nóng (One-hot encoding) của bạn sẽ là [0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , 0].

Mã sau đây thực hiện các chức năng này cho chúng tôi:

```python
os.makedirs('data', exist_ok=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
 path=os.path.join(os.getcwd(), 'data', 'mnist.npz')
)
# Reshaping x_train and x_test tensors so that each image is represented
# as a 1D vector
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
# Standardizing x_train and x_test tensors
x_train = ( 
    x_train - np.mean(x_train, axis=1, keepdims=True)
)/np.std(x_train, axis=1, keepdims=True)
x_test = ( 
    x_test - np.mean(x_test, axis=1, keepdims=True)
)/np.std(x_test, axis=1, keepdims=True)
# One hot encoding y_train and y_test
y_onehot_train = np.zeros((y_train.shape[0], num_labels),
dtype=np.float32)
y_onehot_train[np.arange(y_train.shape[0]), y_train] = 1.0
y_onehot_test = np.zeros((y_test.shape[0], num_labels), dtype=np.float32)
y_onehot_test[np.arange(y_test.shape[0]), y_test] = 1.0
```

Bạn có thể thấy rằng chúng ta đang sử dụng chức năng tf.keras.datasets.mnist.load_data() do TensorFlow cung cấp để tải xuống dữ liệu đào tạo và kiểm tra. Điều này sẽ cung cấp bốn tensors đầu ra

```python
- x_train - Một tensor có kích thước 60000 x 28 x 28 trong đó mỗi hình ảnh là 28 x 28
- y_train - Một vectơ có kích thước 60000, trong đó mỗi phần tử là một nhãn lớp từ 0-9 
- x_test - Tensor có kích thước 10000 x 28 x 28 
- y_test - Vectơ có kích thước 10000
```

Khi dữ liệu được tải xuống, chúng ta định hình lại hình ảnh có kích thước 28 x 28 thành một vectơ 1D. Điều này là do chúng ta sẽ triển khai một mạng lưới thần kinh được kết nối đầy đủ. Các mạng thần kinh được kết nối đầy đủ lấy một vectơ 1D làm đầu vào. Do đó, tất cả các pixel trong hình ảnh sẽ được sắp xếp như một chuỗi các pixel để đưa vào mô hình. Cuối cùng, nếu bạn nhìn vào phạm vi của các giá trị có trong các tensor X_Train và X_Test, chúng sẽ nằm trong phạm vi 0-255 (phạm vi thang độ xám điển hình). Chúng ta sẽ đưa các giá trị này vào phạm vi phương sai đơn vị trung bình bằng không bằng cách trừ trung bình của mỗi hình ảnh và chia cho độ lệch chuẩn.
 
### <font color = 'green'> Triển khai neural network với Keras

Mạng thần kinh được kết nối đầy đủ với 3 lớp có 500, 250 và 10 nút tương ứng. Hai lớp đầu tiên sẽ sử dụng kích hoạt Relu, trong khi lớp cuối cùng sử dụng SoftMax. Để thực hiện điều này, chúng tôi sẽ sử dụng các API KERAS đơn giản nhất có sẵn cho chúng ta - API tuần tự.

```python
model = tf.keras.Sequential([
 tf.keras.layers.Dense(500, activation='relu'),
 tf.keras.layers.Dense(250, activation='relu'),
 tf.keras.layers.Dense(10, activation='softmax')
 ])
```

Bạn có thể thấy rằng tất cả những gì nó cần là một dòng duy nhất trong API tuần tự Keras để xác định mô hình mà chúng ta vừa xác định. Keras cung cấp nhiều loại lớp khác nhau. Bạn có thể thấy danh sách đầy đủ các lớp có sẵn cho bạn tại https://www.tensorflow.org/api_docs/python/tf/keras/layers. Đối với một mạng được kết nối đầy đủ, chúng ta chỉ cần các lớp dày đặc bắt chước các tính toán của một lớp ẩn trong một mạng được kết nối đầy đủ. Với mô hình được xác định, bạn cần biên dịch mô hình này với chức năng tổn thất phù hợp, trình tối ưu hóa và hiệu suất:

```python
optimizer = tf.keras.optimizers.RMSprop()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['acc'])
```

Với mô hình được xác định và biên dịch, giờ đây chúng ta có thể đào tạo mô hình của mình trên dữ liệu đã chuẩn bị.

#### <font color = 'pink'> Training the model

Đào tạo một mô hình không thể dễ dàng hơn với Keras. Khi dữ liệu được chuẩn bị, tất cả những gì bạn cần làm là gọi hàm model.fit () với các đối số cần thiết:

```python
batch_size = 100
num_epochs = 10
train_history = model.fit(
 x=x_train,
 y=y_onehot_train,
 batch_size=batch_size,
 epochs= num_epochs,
 validation_split=0.2
)
```
  
model.fit () chấp nhận một số đối số quan trọng. Chúng ta sẽ đi qua chúng chi tiết hơn ở đây:

- X - Một tenxơ đầu vào. Trong trường hợp của chúng ta, đây là một tenxơ có kích thước 60000 x 784.
- Y - Nhãn được mã hóa một lần nóng (one-hot encoded). Trong trường hợp của chúng ta, đây là một tenxơ có kích thước 60000 x 10.
- batch_size - Các mô hình học tập sâu được đào tạo với các lô dữ liệu (nói cách khác, một cách ngẫu nhiên) trái ngược với việc cung cấp cho bộ dữ liệu đầy đủ cùng một lúc. Kích thước lô xác định có bao nhiêu ví dụ được bao gồm trong một lô. Kích thước lô càng lớn, độ chính xác của mô hình của bạn sẽ càng tốt.

- Epochs - Các mô hình học tập sâu lặp lại thông qua bộ dữ liệu theo các lô nhiều lần. Số lần lặp lại thông qua bộ dữ liệu được gọi là số lượng kỷ nguyên. Trong ví dụ của chúng ta, điều này được đặt thành 10.
- validation_split - Khi đào tạo các mô hình học tập sâu, một bộ xác nhận được sử dụng để theo dõi hiệu suất, trong đó bộ xác thực hoạt động như một proxy cho hiệu suất trong thế giới thực. validation_split xác định số lượng bộ dữ liệu đầy đủ sẽ được sử dụng làm tập hợp con xác thực. Trong ví dụ của chúng ta, điều này được đặt thành 20% tổng kích thước tập dữ liệu

Ở đây, những gì training loss và validation accuracy trông giống như số lượng kỷ nguyên mà chúng ta đã đào tạo mô hình

![](/assets/img/NLP14.png)

Tiếp theo là kiểm tra mô hình của chúng tôi trên một số dữ liệu chưa từng thấy

#### <font color = 'pink'> Kiểm tra model

Kiểm tra mô hình cũng đơn giản. Trong quá trình thử nghiệm, chúng ta đo lường sự mất mát và độ chính xác của mô hình trên bộ dữ liệu thử nghiệm. Để đánh giá mô hình trên bộ dữ liệu, các mô hình Keras cung cấp chức năng thuận tiện gọi là evaluate():

```python
test_res = model.evaluate(
    x=x_test,
    y=y_onehot_test,
    batch_size=batch_size
)
```

Các đối số được mong đợi bởi hàm evaluate() đã được đề cập trong quá trình thảo luận của chúng ta về model.fit ():

- X - một tenxơ đầu vào. Trong trường hợp của chúng ta, đây là một tenxơ có kích thước 10000 x 784. 
- Y - Nhãn được mã hóa một lần nóng. Trong trường hợp của chúng ta, đây là một tensor kích thước 10000 x 10. 
- Batch_size - Kích thước lô xác định số lượng ví dụ được bao gồm trong một lô. Kích thước lô càng lớn thì độ chính xác của mô hình của bạn sẽ càng tốt

Bạn sẽ bị loss 0,138 và độ chính xác là 98%. Bạn sẽ không nhận được các giá trị chính xác giống nhau do sự ngẫu nhiên khác nhau trong mô hình, cũng như trong quá trình đào tạo

# <font color = 'red'> III.Word2vec
  
## <font color = 'yellow'> 1.Giới thiệu

Word2vec là một mô hình đơn giản và nổi tiếng giúp tạo ra các biểu diễn embedding của từ trong một không gian có số chiều thấp hơn nhiều lần so với số từ trong từ điển.

Ý tưởng cơ bản của word2vec có thể được gói gọn trong các ý sau:

- Hai từ xuất hiện trong những văn cảnh giống nhau thường có ý nghĩa gần với nhau.

- Ta có thể đoán được một từ nếu biết các từ xung quanh nó trong câu. Ví dụ, với câu “Hà Nội là … của Việt Nam” thì từ trong dấu ba chấm khả năng cao là “thủ đô”. Với câu hoàn chỉnh “Hà Nội là thủ đô của Việt Nam”, mô hình word2vec sẽ xây dựng ra embeding của các từ sao cho xác suất để từ trong dấu ba chấm là “thủ đô” là cao nhất.

## <font color = 'blue'> 2.Một vài định nghĩa

Trong ví dụ trên đây, từ “thủ đô” đang được xét và được gọi là target word hay từ đích. Những từ xung quanh nó được gọi là context words hay từ ngữ cảnh. Với mỗi từ đích trong một câu của cơ sở dữ liệu, các từ ngữ cảnh được định nghĩa là các từ trong cùng câu có vị trí cách từ đích một khoảng không quá C/2 với C là một số tự nhiên dương. Như vậy, với mỗi từ đích, ta sẽ có một bộ không quá C từ ngữ cảnh.

Xét ví dụ sau đây với câu tiếng Anh: “The quick brown fox jump over the lazy dog” với C=4.

![](/assets/img/NLP15.png)

Khi “the” là từ đích, ta có cặp dữ liệu huấn luyện là (the, quick) và (the, brown). Khi “brown” là từ đích, ta có cặp dữ liệu huấn luyện là (brown, the), (brown, quick), (brown, fox) và (brown, jumps).

Word2vec định nghĩa hai embedding vector cùng chiều cho mỗi từ w trong từ điển. Khi nó là một từ đích, embedding vector của nó là u; khi nó là một từ ngữ cảnh, embedding của nó là v. Sở dĩ ta cần hai embedding khác nhau vì ý nghĩa của từ đó khi nó là từ đích và từ ngữ cảnh là khác nhau. Tương ứng với đó, ta có hai ma trận embedding U và V cho các từ đích và các từ ngữ cảnh.

Có hai cách khác nhau xây dựng mô hình word2vec:

- Skip-gram: Dự đoán những từ ngữ cảnh nếu biết trước từ đích.

- CBOW (Continuous Bag of Words): Dựa vào những từ ngữ cảnh để dự đoán từ đích.

Mỗi cách có những ưu nhược điểm khác nhau và áp dụng với những loại dữ liệu khác nhau.

## <font color = 'blue'> 3.Skip-gram 

Mô hình skip-gram liên tục học bằng cách dự đoán các từ xung quanh được đưa ra một từ hiện tại. Nói cách khác, Mô hình Skip-Gram liên tục dự đoán các từ trong một phạm vi nhất định trước và sau từ hiện tại trong cùng một câu.

skip-gram dự đoán ngữ cảnh hoặc các từ lân cận cho một từ nhất định. Mô hình Skip-Gram được đào tạo trên các cặp n-gram (target_word, context_word) với mã thông báo là 1 và 0. Mã thông báo chỉ định xem context_words đến từ cùng một cửa sổ hay được tạo ngẫu nhiên. Cặp có mã thông báo 0 bị bỏ qua.

### <font color = 'green'> Mã triển khai mô hình Skip-Gram

Các bước cần tuân theo:

- Xây dựng vốn từ vựng corpus
- Xây dựng trình tạo skip-gram [(mục tiêu, ngữ cảnh), mức độ liên quan]
- Xây dựng kiến trúc mô hình skip-gram
- Đào tạo mô hình
- Nhận nhúng Word
 
### <font color = 'green'> 1. Xây dựng vốn từ vựng corpus:

Bước thiết yếu trong khi xây dựng bất kỳ mô hình dựa trên NLP nào là tạo ra một kho tài liệu trong đó chúng tôi trích xuất từng từ duy nhất từ vựng và gán một số nhận dạng duy nhất cho nó.

Kho tư liệu chúng ta đang sử dụng là 'The King James Version of the Bible', từ Dự án Gutenberg, có sẵn miễn phí thông qua mô hình corpus trong nltk.

```python
from nltk.corpus import gutenberg # to get bible corpus
from string import punctuation # to remove punctuation from corpus
import nltk 
import numpy as np
from keras.preprocessing import text
from keras.preprocessing.sequence import skipgrams 
from keras.layers import *
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Model,Sequential 
```
```python
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
# english là ngôn ngữ bạn chọn
stop_words = nltk.corpus.stopwords.words('english')
```

*Quá trình chuyển đổi dữ liệu sang một thứ mà máy tính có thể hiểu được gọi là tiền xử lý. Một trong những hình thức xử lý trước chính là lọc ra những dữ liệu vô dụng. Trong xử lý ngôn ngữ tự nhiên, những từ vô ích (dữ liệu), được gọi là những từ dừng(stop words). Từ dừng là một từ thường được sử dụng (chẳng hạn như “the”, “a”, “an”, “in”) mà công cụ tìm kiếm đã được lập trình để bỏ qua.*

![](/assest/img/NLP16.png)
  
Chúng ta sử dụng chức năng do người dùng xác định để xử lý sơ bộ văn bản giúp loại bỏ các khoảng trắng, chữ số, từ dừng và viết tắt thân văn bản

```python
import re
bible = gutenberg.sents("bible-kjv.txt")
remove_terms = punctuation + '0123456789'
wpt = nltk.WordPunctTokenizer()
def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc,re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc
normalize_corpus = np.vectorize(normalize_document)
```
