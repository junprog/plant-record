import os
import json
from typing import Text
import requests

current_path = os.getcwd()
slack_config_path = os.path.join(current_path, 'notify/slack')
if "slack_webhook_config.py" in os.listdir(slack_config_path):
    import notify.slack.slack_webhook_config as config

def notifySlack(twitter_url=None):
    URL = config.WEB_HOOK_URL

    if twitter_url is not None:
        requests.post(URL, data = json.dumps({
            'text': u"観測しました。\n{}".format(twitter_url),  #通知内容
            'link_names': 1,  #名前をリンク化
        }))