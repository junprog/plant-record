import json
import numpy as np
import labelme.utils as utils

with open('C:/Users/Junya/Desktop/dataset/20210521_184321402_iOS.json', "r",encoding="utf-8") as f:
    dj = json.load(f)

mask_branch = utils.shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][0]['points'], shape_type=None,line_width=1, point_size=1)
mask_pot = utils.shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][1]['points'], shape_type=None,line_width=1, point_size=1)
mask_leaf0 = utils.shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][2]['points'], shape_type=None,line_width=1, point_size=1)
mask_leaf1 = utils.shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][3]['points'], shape_type=None,line_width=1, point_size=1)
mask_leaf2 = utils.shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][4]['points'], shape_type=None,line_width=1, point_size=1)

mask_branch = mask_branch.astype(np.int)[:,:,np.newaxis]*255#booleanを0,1に変換
mask_pot = mask_pot.astype(np.int)[:,:,np.newaxis]*255#booleanを0,1に変換
mask_leaf = mask_leaf0.astype(np.int)[:,:,np.newaxis]*255#booleanを0,1に変換
mask_leaf += mask_leaf1.astype(np.int)[:,:,np.newaxis]*255#booleanを0,1に変換
mask_leaf += mask_leaf2.astype(np.int)[:,:,np.newaxis]*255#booleanを0,1に変換

mask_img = np.concatenate([mask_branch, mask_pot,mask_leaf], axis=2)

#anacondaを使っています
import matplotlib.pyplot as plt

plt.imshow(mask_img)
plt.show()