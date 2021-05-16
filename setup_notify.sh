#/bin/bash

## 追跡対象外のTwitter APIのconfigを作成
TWITTER_CONF_FILE="notify/twitter/twitter_api_config.py"
if [ -e $TWITTER_CONF_FILE ]; then
    echo "Twitter API config file has already exist."
else
    touch $TWITTER_CONF_FILE
    echo "## 追跡対象外\nCONSUMER_KEY = ''\nCONSUMER_SECRET = ''\nACCESS_TOKEN = ''\nACCESS_TOKEN_SECRET = ''" > $TWITTER_CONF_FILE
    echo "Create Twitter API config file."
fi

## 追跡対象外のSlack webhook URLのconfigを作成
SLACK_CONF_FILE="notify/slack/slack_webhook_config.py"
if [ -e $SLACK_CONF_FILE ]; then
    echo "Slack webhook URL config file has already exist."
else
    touch $SLACK_CONF_FILE
    echo "## 追跡対象外\nWEB_HOOK_URL = ''" > $SLACK_CONF_FILE
    echo "Create Slack webhook URL config file."
fi