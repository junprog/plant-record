FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3

# タイムゾーン変更
RUN apt-get -y update && apt install -y tzdata && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV TZ Asia/Tokyo

# ファイルコピー
WORKDIR /app
COPY . /app

### cron, cronで使用するvim, git をインストール
RUN apt-get -y update
RUN apt-get install -y cron \
                     vim \
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