import os
import time
import datetime
import cv2

def captureImage():
    time.sleep(1)
    # Raspy Camera Module
    """
    GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
    """
    #cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)

    # USB Camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)

    ret, img = cap.read()

    # picture filename
    date_info = datetime.datetime.now()
    img_date_info = "{}-{:02}{:02}-{:02}{:02}".format(date_info.year, date_info.month, date_info.day, date_info.hour, date_info.minute)
    img_name = 'pic_{}.jpg'.format(img_date_info)

    current_path = os.getcwd()
    img_dir = os.path.join(current_path, 'images')
    if os.path.isdir(img_dir) == False:
        os.mkdir(img_dir)
    
    img_path = os.path.join(img_dir, img_name)
    cv2.imwrite(img_path, img)
    
    return img_path

if __name__ == "__main__":
    img_path = captureImage()
    print(img_path)
