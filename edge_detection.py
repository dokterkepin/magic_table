import cv2

cap = cv2.VideoCapture("video3.MOV")
ret, frame = cap.read()
if not ret:
    exit()


left_template = frame[250:450, 250:400]
right_template = frame[280:450, 1000:1200]
lw, lh = left_template.shape[1], left_template.shape[0]
rw, rh = right_template.shape[1], right_template.shape[0]


left_gray = cv2.cvtColor(left_template, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_template, cv2.COLOR_BGR2GRAY)
left_canny = cv2.Canny(left_gray, 50, 150)
right_canny = cv2.Canny(right_gray, 50, 150)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_canny = cv2.Canny(frame_gray, 50, 150)

    res_left = cv2.matchTemplate(frame_canny, left_canny, cv2.TM_CCOEFF_NORMED)
    res_right = cv2.matchTemplate(frame_canny, right_canny, cv2.TM_CCOEFF_NORMED)

    _, _, _, left_max_loc = cv2.minMaxLoc(res_left)
    _, _, _, right_max_loc = cv2.minMaxLoc(res_right)

    cv2.rectangle(frame, left_max_loc, (left_max_loc[0] + lw, left_max_loc[1] + lh), (0, 255, 0), 2)
    cv2.rectangle(frame, right_max_loc, (right_max_loc[0] + rw, right_max_loc[1] + rh), (0, 255, 0), 2)

    cv2.imshow("Edge Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
