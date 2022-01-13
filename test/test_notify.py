import sys
import os
current_path = os.getcwd()
sys.path.append(current_path) # /plant-record/ ディレクトリをパスに追加

import notify

twitter_url = notify.tweetInfo(image_filepath="images/pic_2021-0517-0218.jpg")

notify.notifySlack(twitter_url=twitter_url)