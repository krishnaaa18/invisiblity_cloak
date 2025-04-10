import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(2)

# Capture the background
for i in range(30):
    ret, background = cap.read()
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color range (adjusted for better detection)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine both red masks
    mask = mask1 + mask2

    # Clean up the mask to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Invert the mask
    inverse_mask = cv2.bitwise_not(mask)

    # Segment out the red cloak and background
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine the results for final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the output
    cv2.imshow("Invisibility Cloak Effect", final_output)

    # Break if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
