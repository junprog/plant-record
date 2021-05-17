import sys
import os
current_path = os.getcwd()
sys.path.append(current_path)

import get_image as getimg

## 撮影
img_path = getimg.captureImage()