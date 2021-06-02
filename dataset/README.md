# datasetを作るためのディレクトリ

手元のパキラから画像収集 (とりあえずこれを採用)
pythonでGoogleから植物画像をスクレイピング (まだしない)

## メモ
* opencv-python
`cv2.imread() -> [H, W, C]`

* generate_mask.py
    * ground truthのpngファイル作成
```bash
$ python dataset/generate_mask.py [data directory] --labels dataset/labels.txt   
```

実行後のディレクトリ構造:
```
[data directory]
    ├　class_names.txt
    ├　*.jpg
    ├　*.png
    └  SegmentationClassVisualization (optionで作成するかどうか指定可能)
        └  *_vis.jpg
```

* disple_image.py
    * 画像とGround Truthのプロット(generate_maskdata.pyでpng作成した後に使用可能)