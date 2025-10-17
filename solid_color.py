import cv2
import numpy as np

cap = cv2.VideoCapture("video1.mov")
frozen_patch = None

while True: 
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0]) 
    upper_black = np.array([180, 255, 70])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((12, 12), np.uint8)
    noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    noise = cv2.morphologyEx(noise, cv2.MORPH_CLOSE, kernel)
    blur = cv2.GaussianBlur(noise, (5,5), 0)

    contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display = frame.copy()

    if contours:
        fx, fy, fw, fh = 0, 500, 1280, 115
        cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (40, 38, 42), -1)

    cv2.imshow("video", display)
            
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

cap.release()
cv2.destroyAllWindows()
