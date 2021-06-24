# Segmentation

植物の写真 -> segmentation model -> segmentation mask [葉, 枝, 鉢, 背景] 

## データセット

    ディレクトリ構造:

    load_dataset.py
    train/
        ├　jpg, png
    test/
        ├　jpg, png

    参考サイト
    https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/


## 使用モデル

## 学習, 検証結果

学習
```bash
$ python segmentation/train_unet.py --data-dir [dataset directory] --result-dir [result-dir] --gpu
```

