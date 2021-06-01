# datasetを作るためのディレクトリ

手元のパキラから画像収集 (とりあえずこれを採用)
pythonでGoogleから植物画像をスクレイピング (まだしない)

## メモ
* opencv-python
`cv2.imread() -> [H, W, C]`

* generate_maskdata.py
    * ground truthのpngファイル作成

* disple_image.py
    * 画像とGround Truthのプロット(generate_maskdata.pyでpng作成した後に使用可能)