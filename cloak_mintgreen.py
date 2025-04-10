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

    # Light Greenish color range corresponding to hex #ABE4A0
    lower_green = np.array([35, 50, 150])  # Lower HSV bounds for #ABE4A0
    upper_green = np.array([85, 255, 255])  # Upper HSV bounds for #ABE4A0
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up the mask to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Invert the mask
    inverse_mask = cv2.bitwise_not(mask)

    # Segment out the greenish cloak and background
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    # Combine the results for final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the output
    cv2.imshow("Invisibility Cloak Effect (Light Greenish)", final_output)

    # Break if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
