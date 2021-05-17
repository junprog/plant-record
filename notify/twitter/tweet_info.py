import os
import json

from requests_oauthlib import OAuth1Session #OAuthのライブラリの読み込み

current_path = os.getcwd()
twitter_config_path = os.path.join(current_path, 'notify/twitter')
if "twitter_api_config.py" in os.listdir(twitter_config_path):
    import notify.twitter.twitter_api_config as config

def tweetInfo(image_filepath=None):
    CK = config.CONSUMER_KEY
    CS = config.CONSUMER_SECRET
    AT = config.ACCESS_TOKEN
    ATS = config.ACCESS_TOKEN_SECRET
    twitter = OAuth1Session(CK, CS, AT, ATS) #認証処理
    
    if image_filepath is not None:
        url = "https://api.twitter.com/1.1/statuses/update.json"
        url_media = "https://upload.twitter.com/1.1/media/upload.json"

        ## ツイート画像 ##
        files = {"media" : open(image_filepath, 'rb')}
        req_media = twitter.post(url_media, files=files)

        if req_media.status_code != 200:
            print ("画像アップデート失敗: {}".format(req_media.text))
            exit()

        # Media ID を取得
        media_id = json.loads(req_media.text)['media_id']
        print ("Media ID: %d" % media_id)

        ## ツイート文章 ##
        date_info = dict()
        image_filename = os.path.basename(image_filepath)
        date_info["year"] = image_filename[4:8]
        date_info["month"] = image_filename[9:11]
        date_info["day"] = image_filename[11:13]
        date_info["hour"] = image_filename[14:16]
        date_info["minute"] = image_filename[16:18]

        tweet = "{}年 {}月{}日 {}:{} の様子 #PlantRecord".format(
            date_info["year"],
            date_info["month"],
            date_info["day"],
            date_info["hour"],
            date_info["minute"])

        params = {"status" : tweet, "media_ids" : [media_id]}

        res = twitter.post(url, params=params) #post送信

        if res.status_code == 200: #正常投稿出来た場合
            print("Success.")
        else: #正常投稿出来なかった場合
            print("Failed. : %d"% res.status_code)

        return res.text

    else:
        print("画像ファイルが存在しません")
        return None

if __name__ == "__main__":
    tweetInfo(image_filepath="images/pic_2021-0517-0218.jpg")