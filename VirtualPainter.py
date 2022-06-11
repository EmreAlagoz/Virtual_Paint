import cv2 as cv
import numpy as np
import HandTrackingModule as htm

wCam , hCam = 1280,720

vid = cv.VideoCapture(0)
vid.set(3,wCam)
vid.set(4,hCam)

detector = htm.handDetector(min_detection_confidence=0.9)

drawingColor = (0,0,255)
# Previous x and y locations
xp,yp = 0,0

imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    # Image capture
    success , img = vid.read()
    img = cv.flip(img,1)

    # Detecting hands
    img = detector.findHands(img,)
    landmarkList = detector.findPos(img,isDraw=False)
    if len(landmarkList)!=0:
        # Locations of middle and index finger
        x1 , y1 = landmarkList[8][1:]
        x2 , y2 = landmarkList[12][1:]

        # Checking the fingers that up
        fingers = detector.fingersUp(img,isFlipped=True)

        # Decision of selection and drawing mode
        if fingers[0] and fingers[1] and fingers[2]:
            xp, yp = x1, y1
            print('Selection mode')
            cv.rectangle(img,(x1,y1-10),(x2,y2+10),(0,0,255),cv.FILLED)

        if fingers[0] and fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1),10, (0, 0, 255), cv.FILLED)
            print('Drawing mode')
            if (xp,yp) == (0,0):
                xp , yp = x1 , y1
            cv.line(img,(xp,yp),(x1,y1),drawingColor,10)
            cv.line(imgCanvas,(xp,yp),(x1,y1),drawingColor,10)
            xp, yp = x1, y1
        if fingers[0]!=1:
            cv.circle(img, (x1, y1), 30, (0, 0, 0), cv.FILLED)
            print('Deleting mode')
            if (xp, yp) == (0, 0):
                xp, yp = x1, y1
            cv.line(img, (xp, yp), (x1, y1), (0, 0, 0), 30)
            cv.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), 30)
            xp, yp = x1, y1


    imgGray = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)

    cv.imshow('img', img)
    # cv.imshow('canvas', imgCanvas)
    # print(f'Img:{img.shape},canvas:{imgCanvas.shape}')
    if cv.waitKey(1) & 0xFF == ord('q'):
        break