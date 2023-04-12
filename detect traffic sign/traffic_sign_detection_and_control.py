import cv2
import numpy as np
from keras.models import load_model

import time
import control as ctl

cap = cv2.VideoCapture(0)
control = ctl.Control()
traffic = False
# def empty(a):
# pass

# cv2.namedWindow('Parameter')
# cv2.resizeWindow('Parameter', 640, 240)
# cv2.createTrackbar('Threshold1', 'Parameter', 60, 255, empty)
# cv2.createTrackbar('Threshold2', 'Parameter', 160, 255, empty)

while True:
    success, frame = cap.read()
    imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = 60
    threshold2 = 160
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    contour, hrc = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contour == None:
        img_detec = False
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, b, p = cv2.boundingRect(cnt)
            cv2.drawContours(frame, cnt, -1, (255, 255, 0), 6)
            cv2.rectangle(frame, (x - 10, y - 10), (x + b + 10, y + p + 10), (255, 0, 0), 2)
            img_write = frame[y - 10:y + p + 10, x - 10:x + b + 10]
            img = cv2.resize(img_write, (64, 64))
            img = np.array(img)

            # load model
            model = load_model('traffic_sign.h5')
            pred = model.predict(img.reshape(1, 64, 64, 3))
            detec = np.argmax(pred)
            if detec == 0:
                print('Stop')
                cv2.putText(frame, 'STOP', (x - 10, y - 16), 0, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                if y<316 and y>302:
                    control.STOP()
                    traffic = True
                elif y>316 or y<302:
                    traffic = False
                    control.STOP()
            if detec == 1:
                print('Left_turn')
                cv2.putText(frame, 'LEFT_TURN', (x - 10, y - 16), 0, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                if y<316 and y>302:
                    control.STOP()
                    time.sleep(1)
                    control.LEFT()
                    traffic = True
                elif y>316 or y<302:
                    traffic = False
                    control.STOP()
                    control.LEFT()
            if detec == 2:
                print('Right_turn')
                cv2.putText(frame, 'RIGHT_TURN', (x - 10, y - 16), 0, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                if y<316 and y>302:
                    control.STOP()
                    time.sleep(1)
                    control.RIGHT()
                    traffic = True
                elif y>316 or y<302:
                    traffic = False
                    control.STOP()
                    control.RIGHT()

            cv2.imshow('img_write', img_write)

    if traffic == True:
        ctl.STOP()
    else:
        ctl.RUN()
    cv2.imshow('result', frame)
    cv2.imshow('img_write', img_write)
    cv2.waitKey(1)

cv2.destroyAllWindows()