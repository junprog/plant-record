import get_image as getimg
import notify
import segmentation as seg

## 撮影
img_path = getimg.captureImage()

## セグメンテーション
weight_path = 'weights/0615-225649-unet'
_, viz_path = seg.predict(img_path, weight_path)

## twitter 投稿
twitter_url = notify.tweetInfo(image_files=[img_path, viz_path])

## slack 通知
notify.notifySlack(twitter_url=twitter_url)