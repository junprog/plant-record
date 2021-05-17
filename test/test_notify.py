import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加

import json
import notify

twitter_response = notify.tweetInfo(image_filepath="images/pic_2021-0517-0218.jpg")

twitter_response = json.loads(twitter_response)
twitter_url = twitter_response["entities"]["media"][0]["expanded_url"]

notify.notifySlack(twitter_url=twitter_url)