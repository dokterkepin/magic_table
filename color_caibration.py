import cv2
import numpy as np

cap = cv2.VideoCapture("video3.MOV")

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

kernel = np.ones((10, 10), np.uint8) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maze_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    maze_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maze_mask = cv2.bitwise_or(maze_mask1, maze_mask2)
    
    maze_noise = cv2.morphologyEx(maze_mask, cv2.MORPH_CLOSE, kernel)
    maze_noise = cv2.morphologyEx(maze_noise, cv2.MORPH_OPEN, kernel)
    maze_blur = cv2.medianBlur(maze_noise, 5)

    maze_contours, _ = cv2.findContours(maze_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = np.float32([[1, 0, 0], [0, 1, 0]])
    display = frame.copy()

    if maze_contours:
        largest_maze = max(maze_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_maze) > 5000:
            hull = cv2.convexHull(largest_maze)
            rect_maze = cv2.minAreaRect(hull)
            cv2.drawContours(display, [hull], -1, (0, 255, 0), 2)
            (mx, my), (wm, hm), maze_angle = rect_maze

            if wm < hm:
                maze_angle += 90
            
            maze_angle
            if maze_angle > 170:
                maze_angle -= 180 
            
            center = (int(mx), int(my))
            M = cv2.getRotationMatrix2D(center, maze_angle, 1.0) 

    stabilized_frame = cv2.warpAffine(frame, M, (frame_width, frame_height))
    roi = stabilized_frame[90:frame_height-90, 130:frame_width-100]

    cv2.imshow("detection", display)
    cv2.imshow("Magic Table", roi)
    cv2.imshow("Original", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
