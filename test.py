import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(f"Camera index {i} is available")
cap.release()
