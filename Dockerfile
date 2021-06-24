FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3

# タイムゾーン変更
RUN apt-get update && apt install -y tzdata && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV TZ Asia/Tokyo

# file copy
WORKDIR /app
COPY . /app

#RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN apt-get -y update
RUN apt-get install -y cron \
       git 

### pip3 で入らなかったパッケージを apt-get でインストール
RUN apt-get install -y python3-opencv \
                        python3-pyqt5 \
                        python3-matplotlib

### import cv2 できるようにPATHを通す
RUN export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

### その他 パッケージ インストール
RUN pip3 install -q git+https://github.com/tensorflow/examples.git
RUN pip3 install -r requirements.txt

# Dockerfileと同じ階層の"cron.d"フォルダ内にcronの処理スクリプトを格納しておく
#ADD cron.d /etc/cron.d/
#RUN chmod 0644 /etc/cron.d/*

## API key の読み込み