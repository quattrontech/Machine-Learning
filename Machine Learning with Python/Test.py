import cv2
import numpy as np

img = cv2.imread('Photos/green_black_background.png')
# Zoom
rows, cols, channels = img.shape
img = cv2.resize(img, None, fx=0.5, fy=0.5)
rows, cols, channels = img.shape
cv2.imshow('img', img)

# Convert hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([0, 0, 0])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('Mask', mask)

# Corrosion expansion
erode = cv2.erode(mask, None, iterations=1)
cv2.imshow('erode', erode)
dilate = cv2.dilate(erode, None, iterations=1)
cv2.imshow('dilate', dilate)

# Traverse replacement
for i in range(rows):
    for j in range(cols):
        if dilate[i, j] == 0:
            img[i, j] = (255,255,255)  # Replace the color here, which is the BGR channel
cv2.imwrite("Photos/grass_w_code_b_c_changed.png", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
