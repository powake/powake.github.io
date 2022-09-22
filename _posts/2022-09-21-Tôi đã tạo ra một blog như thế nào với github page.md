---
title: Tôi đã tạo ra một blog như thế nào với github page
author: tranlequybao
date: 2022-09-21
categories: [web,blog]
tags: [github page,web,blog]
math: true
mermaid: true
image:
  path: /assets/img/blog_github.png
  width: 800
  height: 500
  alt: Blog - Tạo blog cá nhân với github page.
---
Như tiêu đề bài viết, bài post ghi lại quá trình tôi tạo ra một blog cá nhân với github page và jekyll.

## Sơ lược
### Giới thiệu
Như các bạn đã biết website còn gọi là trang web là tập hợp các trang chứa thông tin bao gồm văn bản, hình ảnh, video,… nằm trên một domain, được lưu trữ trên máy chủ web, ví dụ về website như (vietnix.vn, google.com, facebook.com).Website được xem là công cụ hỗ trợ đắc lực cho hoạt động Marketing online, góp phần quảng bá rộng rãi hình ảnh doanh nghiệp, quảng cáo sản phẩm, dịch vụ đến khách hàng nhanh chóng giúp xây dựng thương hiệu, tạo dựng sự uy tín, đồng thời nâng cao sức mạnh cạnh tranh cho các đơn vị kinh doanh trên thị trường.
Hiện tại có rất nhiều cách để tạo 1 trang web (cho người không biết lập trình cho đến các ;lập trình viên chuyên nghiệp) như Wordpress,Wix,Python django, Ruby on rails ... Trong bài viết này, tôi muốn giới thiệu đến các bạn một cách đơn giản hơn (cho các bạn biết một tý về lập trình) đó là blog với github page và jekyll.
### Mục tiêu bài viết
  * Cung cấp kiến thức đầy đủ về việc thiết lập cơ bản cho đến tạo ra một web với đầy đủ chức năng.
  * Ghi lại quá trình làm việc và học tập trên nền tảng này.
### Các liên kết tham khảo
  * [https://pages.github.com/](https://pages.github.com/)
  * [https://viblo.asia/p/dung-jekyll-travis-va-github-pages-de-tao-ra-muon-van-trang-web-de-dang-Qbq5QWp3ZD8](https://viblo.asia/p/dung-jekyll-travis-va-github-pages-de-tao-ra-muon-van-trang-web-de-dang-Qbq5QWp3ZD8)
  * [https://github.com/cotes2020/chirpy-starter](https://github.com/cotes2020/chirpy-starter)
## Tạo blog
Với các trang GitHub, GitHub cho phép bạn lưu trữ một trang web từ kho lưu trữ của mình.
### Khởi tạo repositories
* Truy cập vào [https://github.com/topics/jekyll-plugin](https://github.com/topics/jekyll-plugin) và chọn một template bạn yêu thích từ jekelyy. Trong bài này tôi chọn template [https://github.com/cotes2020/chirpy-starter](https://github.com/cotes2020/chirpy-starter).
* Fork repositories và đặt tên lại <tên blog của bạn>.github.io
    *Fork repositories từ chirpy-starter.
    
    ![image1](/assets/img/Screenshot from 2022-09-21 10-51-54.png)
    
    *Đặt tên lại repositpries.
    
    ![image2](/assets/img/Screenshot from 2022-09-21 10-52-20.png)
    
    *Kết quả.
    
    ![image3](/assets/img/Screenshot from 2022-09-21 10-52-41.png)
    
### Cài đặt
#### Ubuntu
* Cài đặt Ruby và các điều kiện tiên quyết khác :
  ```shell
  sudo apt-get install ruby-full build-essential zlib1g-dev
  ```
  
  ![image4](/assets/img/Screenshot from 2022-09-22 08-45-16.png)
  
  ```shell
  echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
  echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
  echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
  ```
  
  ![image5](/assets/img/Screenshot from 2022-09-22 08-45-29.png)
  
  ```shell
  gem install jekyll bundler
  ```
  ![image6](/assets/img/Screenshot from 2022-09-22 08-45-29.png)
  
### Config code
* Clone source code về máy để config
  ```shell
  git clone https://github.com/tlqbao/tlqbao.github.io.git
  ```
  ```shell
  cd tlqbao.github.io.git
  ```
  * Cài đặt các gói phụ thuộc
  ```shell
  bundler
  ```
  ![image7](/assets/img/Screenshot from 2022-09-22 08-48-48.png)

* Run local host
  ```shell
  bundle exec jekyll s
  ```
  ![image8](/assets/img/Screenshot from 2022-09-22 08-50-58.png)

  ![image9](/assets/img/Screenshot from 2022-09-22 08-52-15.png)

* Bạn sẽ tìm thấy hướng dẫn cụ thể cho việc config template này ở [https://github.com/cotes2020/jekyll-theme-chirpy#documentation](https://github.com/cotes2020/jekyll-theme-chirpy#documentation)

  Trong bài viết này sẽ chỉ đề cập đến việc config những phần cơ bản, bạn có thể xem và config sâu hơn tùy ý theo tài liệu.
  Tất cả các phần tùy chỉnh đều nằm ở file _config.yml
* Title  (line 20).

    title: <tên blog của bạn>
    ví dụ: 
    
      title: tlqbao 
      
* url (line 32)

    url: '<tên repositeries>'
    
    ví dụ: 
    
      url: 'https://tlqbao.github.io' 
      
* timezone (line 19)

  timezone: Asia/Ho_Chi_Minh
  
* tagline (line 26)

  tagline : <câu nói yêu thích của bạn hây bất cứ thứ gì đó ...> 
  

### Deploy blog
* Tạo commit và push code
  ```shell
  git add .
  git status
  git commit -m "first config"
  git push origin main
  ```
  ![image10](/assets/img/Screenshot from 2022-09-22 08-54-07.png)
  
* Config git.

  Chọn **Settings ---> **Pages .
  
  ![image11](/assets/img/Screenshot from 2022-09-22 08-56-27.png)
  
  Ở **Deploy from branch chọn **Github Action .
  
  ![image12](/assets/img/Screenshot from 2022-09-22 08-59-29.png)
  
  Ở **Deploy from branch chọn **gh-pages.
  
  ![image13](/assets/img/Screenshot from 2022-09-22 10-17-04.png)
  ![image14](/assets/img/Screenshot from 2022-09-22 10-17-22.png)
  
  Kết quả: 
  
  ![image15](/assets/img/Screenshot from 2022-09-22 10-17-58.png)
  
### Thêm miền tùy chỉnh
#### Có SSL
* Domain porkbun
#### Không SSL
* Domain venom

### Tạo bài viết đầu tiên
#### Giới thiệu ngôn ngữ Maskdown
Một số syntax cơ bản trong Markdown.
Markdown	HTML	Kết quả
| Markdown | HTML | Kết quả  |
|:-------|:------:|-------:|
|  # Header 1  |  <h1>Header 1</h1>  |   Header 1  |
|  ## Header 2  |  <h2>Header 2</h2>  |   Header 2 |
|  ### Header 3  |  <h3>Header 3</h3>  |   Header 3  |
|  #### Header 4  |  <h4>Header 4</h4>  |   Header 4  |
| ====== | ====== | =====: |
| Footer | Footer | Footer |

#### Các bước tạo bài viết
## Các lỗi thường gặp và cách fix
* Lỗi abc 
* Lỗi xyz 

## Tương lai
* Bài viết sẽ liên được cập nhật thường xuyên.
