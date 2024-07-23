import time
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1) # 0 sets to default camera (index)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
pTime = 0
while True:
    # reads next frame
    ret, frame = cap.read()
    
    # frame read correctly -> ret is True
    if not ret:
        print("Can't receive frame. Exiting...")
        break
    
    # cvtColor converts an image from one colour space to another
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    grey = cv.flip(grey, flipCode=1)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(grey, f'FPS:{int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # display the frame --- cv.imshow(winname, mat)
    cv.imshow('frame', grey)
    
    # wait for...
    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()