---
title: Python Django | Tạo webapp với framework Django
author: tranlequybao
date: 2022-09-17
categories: [Python,Django]
tags: [django,web]
math: true
mermaid: true
image:
  path: /assets/img/django.png
  width: 800
  height: 500
  alt: Django - framework for web development.
---
Một bài viết đầy đủ từ tạo, config, nâng cấp và deploy web lên server trên các nền tảng khác nhau.

## Sơ lược 
### Giới thiệu framework django

  * Django là khung ứng dụng web cho Python 
  * Django mới nhất hỗ trợ dòng python 3
  
### Mục tiêu bài viết 

  * Cung cấp kiến thức đầy đủ về việc thiết lập cơ bản cho đến tạo ra một web cơ bản với đầy đủ chức năng.
  * Ghi lại kinh nghiệm hơn 1 năm làm việc và học tập trên nền tảng này.
  
### Các liên kết và nguồn tài liệu tham khảo

  Tài liệu từ Django
  
  * [https://docs.djangoproject.com/en/4.0/](https://docs.djangoproject.com/en/4.0/)
  
  * [https://www.geeksforgeeks.org/django-tutorial/](https://www.geeksforgeeks.org/django-tutorial/)
  
  * [https://tomomano.gitlab.io/intro-aws/#aws_account](https://tomomano.gitlab.io/intro-aws/#aws_account)
  
## Cài đặt
### Khởi tạo môi trường ảo

  Việc khởi tạo môi trường ảo để phục vụ cho project là cần thiết, tránh sự ảnh hưởng từ môi trường máy đến project và ngược lại.
  Trong bài viết này chỉ đề cập đến môi trường ảo trên ubuntu (trên các hệ điều hành khác sẽ không được đề cập tại đây)
  
  Cài đặt thư viện python3-venv
  ```shell
  sudo apt-get install -y python3-venv
  ```
  Tạo folder dự án
  ```shell
  mkdir my_project
  ```
  Tạo môi trường ảo
  ```shell
  cd my_project
  ```
  ```shell
  python -m venv django-env
  ```
  Kích hoặt môi trường ảo 
  ```shell
  source django-env/bin/activate 
  ```
  Hủy kích hoạt 
  ```shell
  deactivate 
  ```
### Cài đặt các thư viện yêu cầu 

  Để thuận tiện cho quá trình thêm và cài đặt các thư viện yêu cầu cần thiết trong dự trên venv chúng ta nên tạo file requirements.txt. Ở đây chúng ta    không sử dụng "pip freeze" tránh trường hợp cài quá nhiều thư viện không cần thiết.
  ```shell
  touch requirements.txt 
  ```
  ```shell
  pip install -r requirements.txt 
  ```
## Tạo ứng dụng web django đầu tiên
### Tạo project 
  ```shell
  django-admin startproject web_project
  ```
### Tạo web app bên trong project
  ```shell
  cd web_project
  ```
  ```shell
  python manage.py startapp my_web
  ```
  
### Chạy thử nghiệm
  ```shell
  python manage.py runserver
  ```
  
### Tìm hiểu về mô hình MVT (Models  ---> Views ---> Templates)
### Tùy chỉnh cơ bản 
### Áp dụng 
## Tạo blog cá nhân 
### Tạo webapp django 
### Thiết kế Frontend với Nicepage (hoặc bất kỳ phần thứ gì bạn thích)
### Kết hợp django với templates từ nicepage 
## Deloy 
### AWS 
### Heroku 
## Nâng cấp blog 
### Thêm gói ckeditor vào blog
### Thêm slug và tagit vào django 
### Thêm cloudinary lưu trữ bên ngoài heroku 
## Thêm miền tùy chỉnh  
### Tạo tên miền .tk miễn phí
### Thiết lập tên miền miễn phí và heroku 
## Những lỗi thường gặp và cách sửa
## Tương lai 
* Bài viết sẽ luôn được cập nhật thường xuyên.

