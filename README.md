# plant-record
植物の成長記録を全自動で行おう。成長差分の認識もしたいな。成長の統計の可視化もしたいな。目指せ publish

## Environment

- jetson nano (Ubuntu 16.04 LTS)
- python 3.6
    - opencv-python
    - (pytorch)
    - requests (twitter api用)
    - requests-oauthlib (twitter api用)
- USB Web Camera (RasberryPi Camera Moduleは画質、色彩が悪かった)

## Features

- [getImage](#getImage)
- [notify](#notify)
- [segmentation](#segmentation)
- [recognizeDiff](#recognizeDiff)
- [GUI](#GUI)

<a id="getImage"></a>

### getImage

1日に1,2枚植物画像を取得する。

- 定期処理  : cron
- 画像取得  : python + opencv-python

<a id="notify"></a>

### notify

取得画像を通知する。

- 通知      : Twitter, Slack + incoming-webhook

```bash
$ sh setup_notify_twitter.sh # config 作成 → twitter, slackのconfigを各自入力
$ python3 notify_test.py
```

<a id="segmentation"></a>

### segmentation

随時更新

<a id="recognizeDiff"></a>

### recognizeDiff

随時更新

<a id="GUI"></a>

### GUI

随時更新
