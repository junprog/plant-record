import os
import time
import datetime
import cv2

def main():
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
    cap = cv2.VideoCapture(1, cv2.CAP_V4L)

    ret, img = cap.read()

    # picture filename
    path = os.getcwd()
    date_info = datetime.datetime.now()
    pic_name = 'pic_{}.jpg'.format(date_info.isoformat()[0:16])
    pic_name = os.path.join(path, 'images', pic_name)

    cv2.imwrite(pic_name, img)
    del(cap)

if __name__ == "__main__":
    main()
    print('Done')
