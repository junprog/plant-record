
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
#import tensorflow as tf
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLCDNumber, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QFileDialog
import datetime
from PyQt5.QtCore import QTimer
import random
import matplotlib.pyplot as plt

image_path1 = "./E14h481VcAUUa1t.jpg"
image_path2 = "./highpass.png"

"""
画像を入力、ground-truth, 予測結果の横並びにする
"""

"""
### info_hbox: 基本情報をグループ化
- lbl_temp:     気温情報
- lbl_humidity: 湿度情報
- lbl_date:     日時
- date_timer:   時計

### load_image_hbox: モデル読み込み関係をグループ化
- btn_model:      モデルを読み込むボタン (エクスプローラーを開く)
- self.textbox:   読み込んだファイル名を表示
- addStretch(1):  

### inputs_vbox: 入力画像周りをグループ化
- self.img_input:       入力画像
- self.lbl_input_name:  入力画像名 (画像サイズ)
- btn_input:            入力画像を変更するボタン
- addStretch(1):        ボタンが不用意に伸びないようにする

### outputs_vbox: 出力周りをグループ化
- img_output:             出力画像
- self.lbl_output_name:   出力画像名 (画像サイズ)
- btn_inference_hbox:       
　- self.btn_inference:    推論処理を行う
　- addStretch(1)
- addStretch(1)

### in_out_images_hbox: 入出力画像を横並びにする
- inputs_vbox:    入力画像周り
- outputs_vbox:   出力画像回り

QWidget
- info_hbox
- load_image_hbox
- images_net
- addStretch(1)
"""

class Example(QWidget):

    def __init__(self):
        
        super().__init__()

        self.update_interval = 1000     # 温度、湿度、日時を更新する間隔 [ms]
        self.fontsize = 15              # 上記の情報の文字サイズ

        self.temp_log = []

        self.initUI()


    def initUI(self):     
        #　ウインドウサイズ (x座標, y座標, 横幅, 縦幅)
        self.setGeometry(300, 300, 800, 500)


        ### 基本情報
        # 気温
        self.lbl_temp = QLabel('')
        self.lbl_temp.setFont(QFont('Arial', self.fontsize))
        # 湿度
        self.lbl_humidity = QLabel('')
        self.lbl_humidity.setFont(QFont('Arial', self.fontsize))
        
        ### 学習済みモデルを読み込む
        btn_log = QPushButton('log')             # モデルを読み込むボタン
        btn_log.resize(btn_log.sizeHint())      # ボタンのサイズの自動設定
        btn_log.clicked.connect(self.plot_log)  # ボタンおされた時の処理
        
        # 日時
        self.lbl_date = QLabel('')
        self.lbl_date.setFont(QFont('Arial', self.fontsize))

        #一秒間隔で基本情報の更新
        self.date_timer = QTimer(self)
        self.date_timer.timeout.connect(self.update_date)
        self.date_timer.timeout.connect(self.update_temp)
        self.date_timer.timeout.connect(self.update_lbl_humidity)
        self.date_timer.start(self.update_interval)

        # 時計
        timer = QTimer(self)        
        timer.timeout.connect(self.updtTime)
        self.testTimeDisplay = QLCDNumber(self)
        self.testTimeDisplay.setSegmentStyle(QLCDNumber.Filled)
        self.testTimeDisplay.setDigitCount(8)
        self.testTimeDisplay.resize(200, 200)
        self.updtTime()
        timer.start(1000)

        # 基本情報をまとめる
        info_hbox = QHBoxLayout()
        info_hbox.addWidget(self.lbl_temp)
        info_hbox.addWidget(self.lbl_humidity)
        info_hbox.addWidget(btn_log)
        info_hbox.addStretch(1)
        info_hbox.addWidget(self.lbl_date)
        info_hbox.addWidget( self.testTimeDisplay )




        ### 学習済みモデルを読み込む
        btn_model = QPushButton('File')             # モデルを読み込むボタン
        btn_model.resize(btn_model.sizeHint())      # ボタンのサイズの自動設定
        btn_model.clicked.connect(self.load_model)  # ボタンおされた時の処理
        # 読み込んだモデル名を表示
        self.textbox = QLabel() # テキストボックス
        # モデル読み込み関係をまとめる
        load_image_hbox = QHBoxLayout()
        load_image_hbox.addWidget(btn_model)
        load_image_hbox.addWidget(self.textbox)
        load_image_hbox.addStretch(1)



        ### 入力画像用
        # 画像名 (画像サイズ) 
        self.lbl_input_name = QLabel()
        # 入力画像
        self.img_input = QLabel()
        self.setImage(self.img_input, self.lbl_input_name, image_path1, name="Input Image")     # 画像のセット
        # 入力画像読み込みのボタン
        btn_in = QPushButton('Load Input image')        # ボタンウィジェット作成
        btn_in.resize(btn_in.sizeHint())                # ボタンのサイズの自動設定
        btn_in.clicked.connect(self.load_input_image)   # ボタンが押された時の処理
        # ボタンが伸びないようにする
        btn_input = QHBoxLayout()
        btn_input.addWidget(btn_in)
        btn_input.addStretch(1)
        # 入力画像関連をinputsにまとめる
        inputs_vbox = QVBoxLayout()
        inputs_vbox.addWidget(self.img_input)
        inputs_vbox.addWidget(self.lbl_input_name)
        inputs_vbox.addLayout(btn_input)
        inputs_vbox.addStretch(1)


        ### 推論画像用
        # 画像名
        self.output_name = QLabel( )
        # 出力画像用 (入力画像が読み込まれたら仮で入力画像と同じものをセット)
        self.img_output = QLabel(self)
        self.setImage(self.img_output, self.output_name, image_path1, name="Output Image") # 出力画像をセット (仮)
        # 推論を行う
        self.btn_inference = QPushButton('Inference')       # 推論を行うボタン (モデルの読み込み状態に合わせてボタンを押せなくする)
        self.btn_inference.resize(btn_in.sizeHint())        # ボタンのサイズの自動設定
        self.btn_inference.setEnabled(False)                # 初期状態ではボタンを押せないようにする
        self.btn_inference.clicked.connect(self.Inference)  # ボタンが押された時の処理

        # ボタンが伸びないようにする
        btn_inference_hbox = QHBoxLayout()
        btn_inference_hbox.addWidget(self.btn_inference)
        btn_inference_hbox.addStretch(1)

        # 出力画像関連をまとめる
        outputs_vbox = QVBoxLayout()
        outputs_vbox.addWidget(self.img_output)
        outputs_vbox.addWidget(self.output_name)
        outputs_vbox.addLayout(btn_inference_hbox)
        outputs_vbox.addStretch(1)

        ### 推論画像用
        ### 入出力画像を横並びにする
        images_net = QHBoxLayout()
        images_net.addLayout(inputs_vbox)
        images_net.addLayout(outputs_vbox)

        ### 全体の構造
        main = QVBoxLayout()
        main.addLayout( info_hbox )
        main.addLayout( load_image_hbox )
        main.addLayout( images_net )
        main.addStretch(1)
        self.setLayout(main)
        self.show()




    # 時間の更新
    def updtTime(self):
        currentTime = QDateTime.currentDateTime().toString('hh:mm:ss')
        self.testTimeDisplay.display(currentTime)

    # 日付の更新
    def update_date(self):
        timestamp = str( "{:%Y/%m/%d}".format( datetime.datetime.now()  ) )
        self.lbl_date.setText( timestamp )

    # 温度の更新
    def update_temp(self):
        rand = random.random()
        self.temp_log.append(rand)
        self.lbl_temp.setText( "温度 " + str(rand) + "[℃]" )
    
    # 湿度の更新
    def update_lbl_humidity(self):
        rand = random.random()
        self.lbl_humidity.setText( "湿度 " + str(rand) + "[%]" )
        
    def plot_log(self):
        # 現時点のlogをプロット
        plt.figure()
        plt.plot(self.temp_log)
        plt.title("temperature log")
        plt.xlabel("")
        plt.ylabel("℃")
        plt.show()
        # csvに保存
        if not os.path.exists("./log"):
            os.makedirs("./log")        
        # csvに保存
        filename = "".join( [ "./log/log_", str( "{:%m%d%s}".format( datetime.datetime.now() ) ), ".csv" ] )
#        filename = "./log/log_{}.csv"
        with open(filename, "w", newline="") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow( self.temp_log )

    # クリックのテスト用
    def onClick(self):
        print("click")

    # 画像のセット
    def setImage(self, lbl_fig, lbl_name, image_path, name=""):
        pixmap = QPixmap(image_path)
       # 画像の配置
        lbl_fig.setPixmap(pixmap)
        # テキストの更新
        lbl_name.setText( ( '%s (%dx%d)' % (name, pixmap.height(), pixmap.width()) ) )
        
    # モデルを読み込むボタンを押したときの処理
    def load_model(self):
        # 第二引数はダイアログのタイトル、第三引数は表示するパス
        fname = QFileDialog.getOpenFileName(self, 'Open file', '')
        # ファイル名
        name = fname[0].split("/")[-1]
        # テキストを読み込んだファイル名に更新
        self.textbox.setText(name)   
        if name.split(".")[-1] == "h5":  # ほかのフォーマットも読み込むなら追加
            # モデルの読み込み
            self.model = tf.keras.models.load_model(fname[0], compile=False)
            # ボタン「"infenrece"」を押せるようにする
            self.btn_inference.setEnabled(True)
            print(name, "Load")
        else:
            self.btn_inference.setEnabled(False)



    # エクスプローラーからモデルを読み込む
    def load_input_image(self):
        # 第二引数はダイアログのタイトル、第三引数は表示するパス
        fname = QFileDialog.getOpenFileName(self, 'Open file', '')
        # 画像のセット
        fig_format = fname[0].split(".")[-1]
        if ( fig_format == "jpg" ) or ( fig_format == "png" ):  # 他の拡張子も入れるなら追加
            self.setImage(self.img_input,  self.lbl_input_name, fname[0], name="Input Image")
            self.setImage(self.img_output, self.lbl_input_name, fname[0], name="Output Image")


    # 更新予定
    def Inference(self):
        print("Hello World.")
        pass
    
if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())



