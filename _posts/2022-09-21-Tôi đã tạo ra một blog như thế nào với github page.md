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
  ```shell
  echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
  echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
  echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
  ```
  ```shell
  gem install jekyll bundler
  ```
### Config code
### Deploy blog
### Thêm miền tùy chỉnh
#### Có SSL
#### Không SSL
### Tạo bài viết đầu tiên
#### Giới thiệu ngôn ngữ Maskdown
#### Các bước tạo bài viết
## Các lỗi thường gặp và cách fix
* Lỗi abc 
* Lỗi xyz 
## Tương lai
* Bài viết sẽ liên được cập nhật thường xuyên.
