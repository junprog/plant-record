import json
import numpy as np
import labelme.utils as utils

with open('../images/E14h481VcAUUa1t.json', "r",encoding="utf-8") as f:
    dj = json.load(f)

mask_branch = utils.shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][0]['points'], shape_type=None,line_width=1, point_size=1)
mask_pot = utils.shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][1]['points'], shape_type=None,line_width=1, point_size=1)
mask_leaf = utils.shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj['shapes'][2]['points'], shape_type=None,line_width=1, point_size=1)

mask_branch = mask_branch.astype(np.int)[:,:,np.newaxis]*255#booleanを0,1に変換
mask_pot = mask_pot.astype(np.int)[:,:,np.newaxis]*255#booleanを0,1に変換
mask_leaf = mask_leaf.astype(np.int)[:,:,np.newaxis]*255#booleanを0,1に変換

mask_img = np.concatenate([mask_branch, mask_pot,mask_leaf], axis=2)

#anacondaを使っています
import matplotlib.pyplot as plt

plt.imshow(mask_img)
plt.show()