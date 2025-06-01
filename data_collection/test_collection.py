import cv2
import time

camera_indices = []
windows = {}
caps = {}

# Try to open all camera indices from 0 to 9
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            camera_indices.append(i)
            caps[i] = cap
            window_name = f"Camera {i}"
            windows[i] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
            cv2.putText(frame, f"Camera Index: {i}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(window_name, frame)
        else:
            cap.release()
    else:
        cap.release()

print(f"Available camera indices: {camera_indices}")

# Wait until user presses a key
print("Press any key in any window or 'q' to exit.")
while True:
    key = cv2.waitKey(1)
    if key != -1:
        break
    for i in camera_indices:
        ret, frame = caps[i].read()
        if ret:
            cv2.putText(frame, f"Camera Index: {i}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(windows[i], frame)

# Clean up
for i in camera_indices:
    caps[i].release()
cv2.destroyAllWindows()
