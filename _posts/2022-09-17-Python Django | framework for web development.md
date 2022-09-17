---
title: Python Django | framework for web development
author: tranlequybao
date: 2022-09-17
categories: [Python,Django]
tags: [django,web]
math: true
mermaid: true
---
Một bài viết đầy đủ từ tạo, config, nâng cấp và deploy web lên server trên các nền tảng khác nhau.

## 1/ Sơ lược 

#### 1-1/ Giới thiệu framework django

  * Django là khung ứng dụng web cho Python 
  * Django mới nhất hỗ trợ dòng python 3
  
#### 1-2/ Mục tiêu bài viết 

  * Cung cấp kiến thức đầy đủ về việc thiết lập cơ bản cho đến tạo ra một web cơ bản với đầy đủ chức năng.
  * Ghi lại kinh nghiệm hơn 1 năm làm việc và học tập trên nền tảng này.
  
#### 1-3/ Các liên kết và nguồn tài liệu tham khảo

  Tài liệu từ Django
  
  * https://docs.djangoproject.com/en/4.0/
  
  * https://www.geeksforgeeks.org/django-tutorial/
  
  * https://tomomano.gitlab.io/intro-aws/#aws_account
  
## 2/ Cài đặt

#### 2-1/ Khởi tạo môi trường ảo

  Việc khởi tạo môi trường ảo để phục vụ cho project là cần thiết, tránh sự ảnh hưởng từ môi trường máy đến project và ngược lại.
  Trong bài viết này chỉ đề cập đến môi trường ảo trên ubuntu (trên các hệ điều hành khác sẽ không được đề cập tại đây)
  
  Cài đặt thư viện python3-venv
  ```python
  sudo apt-get install -y python3-venv
  ```
  Tạo folder dự án
  ```python
  mkdir my_project
  ```
  Tạo môi trường ảo
  ```python
  cd my_project
  ```
  ```python
  python -m venv django-env
  ```
  Kích hoặt môi trường ảo 
  ```python
  source django-env/bin/activate 
  ```
  Hủy kích hoạt 
  ```python
  deactivate 
  ```
#### 2-2/ Cài đặt các thư viện yêu cầu 

  Để thuận tiện cho quá trình thêm và cài đặt các thư viện yêu cầu cần thiết trong dự trên venv chúng ta nên tạo file requirements.txt. Ở đây chúng ta    không sử dụng "pip freeze" tránh trường hợp cài quá nhiều thư viện không cần thiết.
  ```python
  touch requirements.txt 
  ```
  ```python
  pip install -r requirements.txt 
  ```
## 3/ Tạo ứng dụng web django đầu tiên

#### 3-1/ Tạo project 
  ```python
  django-admin startproject web_project
  ```
#### 3-2/ Tạo web app bên trong project
  ```python
  cd web_project
  ```
  ```python
  python manage.py startapp my_web
  ```
  
#### 3-3/ Chạy thử nghiệm
  ```python
  python manage.py runserver
  ```
  
#### 3-4/ Tìm hiểu về mô hình MVT (Models  ---> Views ---> Templates)

#### 3-5/ Tùy chỉnh cơ bản 

#### 3-6/ Áp dụng 

## 4/ Tạo blog cá nhân 

#### 4-1/ Tạo webapp django 

#### 4-2/ Thiết kế Frontend với Nicepage (hoặc bất kỳ phần thứ gì bạn thích)

#### 4-3/ Kết hợp django với templates từ nicepage 

## 5/ Deloy 

#### 5-1/ AWS 

#### 5-2/ Heroku 

## 6/ Nâng cấp blog 

#### 6-1/ Thêm gói ckeditor vào blog

#### 6-2/ Thêm slug và tagit vào django 

#### 6-3/ Thêm cloudinary lưu trữ bên ngoài heroku 

## 7/ Thêm miền tùy chỉnh  

#### 7-1/ Tạo tên miền .tk miễn phí

#### 7-2/ Thiết lập tên miền miễn phí và heroku 

## 8/ Những lỗi thường gặp và cách sửa

## 9/ Tương lai 

