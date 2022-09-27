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
