FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

# タイムゾーン変更
RUN apt-get update && apt install -y tzdata && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV TZ Asia/Tokyo

# file copy
WORKDIR /app
COPY . /app

RUN apt-get -y update
RUN apt-get install -y cron \
        python3-opencv
RUN pip3 install -r requirements.txt

### RUN /etc/init.d/cron start ## エラー でる？

# Dockerfileと同じ階層の"cron.d"フォルダ内にcronの処理スクリプトを格納しておく
ADD cron.d /etc/cron.d/
RUN chmod 0644 /etc/cron.d/*

## cron に設定する
## time zone を Asia/Tokyo にする必要あり?

## API key の読み込み