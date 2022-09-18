---
title: Python Kivy | Tạo ứng dụng điện thoại thông minh với Kivy và Buildozer
author: tranlequybao
date: 2022-09-18
categories: [Python,Kivy]
tags: [kivy,mobile app,buildozer]
math: true
mermaid: true
image:
  path: /assets/img/kivy.png
  width: 800
  height: 500
  alt: Kivy - framework for mobile app development.
---
Trong bài viết này, chúng ta cùng tìm hiểu về framework Kivy, cách xây dựng một ứng dụng điện thoại với Kivy và build thành app android với buildozer.

## Sơ lược
### Giới thiệu framework django
### Mục tiêu bài viết
### Các liên kết và nguồn tài liệu tham khảo 
## Cài đặt
### Khởi tạo môi trường
### Cài đặt các thư viện yêu cầu
## Ứng dụng đầu tiên
## Ứng dụng nâng cao
## Khởi tạo môi trường build trên Ubuntu 20.04
### Các thư viện yêu cầu
  Thư viện cần thiết 
  ```shell
  sudo apt-get install python3-distutils

  sudo python3 get-pip.py

  sudo apt-get install -y python3-pip build-essential git python3 python3-dev

  sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev zlib1g-dev

  sudo apt-get install cython

  sudo pip3 install kivy

  python3 main.py - code is available

  sudo apt-get install libltdl-dev libffi-dev libssl-dev autoconf autotools-dev

  sudo apt install -y git zip unzip openjdk-8-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev pip3 install --user --upgrade Cython==0.29.19 virtualenv # the --user should be removed if you do this in a venv
  
  ```
  Thêm dòng sau vào cuối tệp ~ / .bashrc của bạn
  ```shell
  export PATH=$PATH:~/.local/bin/
  ```
  Clone source code buildozer
  ```shell
  git clone https://github.com/kivy/buildozer.git
  cd buildozer
  sudo python3 setup.py install
  buildozer init
  buildozer android debug
  ``` 
## Build app adroid 
## Các lỗi thường gặp và cách sửa
## Tương lai 
*Bài viết sẽ được cập nhật thường xuyên.
