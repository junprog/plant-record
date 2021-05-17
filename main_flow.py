import json

import get_image as getimg
import notify

## 撮影
img_path = getimg.captureImage()

## twitter 投稿
twitter_response = notify.tweetInfo(image_filepath=img_path)
twitter_response = json.loads(twitter_response)
twitter_url = twitter_response["entities"]["media"][0]["expanded_url"]

## slack 通知
notify.notifySlack(twitter_url=twitter_url)