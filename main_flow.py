import get_image as getimg
import notify

## 撮影
img_path = getimg.captureImage()

## twitter 投稿
twitter_url = notify.tweetInfo(image_filepath=img_path)

## slack 通知
notify.notifySlack(twitter_url=twitter_url)