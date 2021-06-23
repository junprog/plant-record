import requests
import random
import shutil
import bs4
import ssl
import os
import datetime
ssl._create_default_https_context = ssl._create_unverified_context

def image(data):
    Res = requests.get("https://www.google.com/search?hl=jp&q=" + data + "&btnG=Google+Search&tbs=0&safe=off&tbm=isch")
    Html = Res.text
    Soup = bs4.BeautifulSoup(Html,'lxml')
    links = Soup.find_all("img")
    link = random.choice(links).get("src")
    return link

def download_img(url, file_name):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name+".png", 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


if __name__ == "__main__":
    # 画像を保存するディレクトリimagesを作成
    if not os.path.exists("./images"):
        os.makedirs("images")

    num = input("検索回数:")
    data = input("検索ワード:")

    for i in range( int( num ) ):   # num個画像を保存
        link = image(data)
        # 保存するファイル名を時間情報から付ける
        dt_now = datetime.datetime.now()
        filename = "./images/" + dt_now.strftime( '%m%d%H%M%S%f' )
        # 画像をダウンロードする
        download_img(link, filename)
        print("%d: %s" % ( i+1, filename ) )
    print("OK")