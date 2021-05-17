FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

# file copy
WORKDIR /workspace
COPY . /workspace

RUN apt-get -y update
RUN apt-get install -y cron
RUN apt-get install -y python3-opencv
RUN pip3 install -r requirements.txt

RUN /etc/init.d/cron start
## cron に設定する
## time zone を Asia/Tokyo にする必要あり?

## API key の読み込み