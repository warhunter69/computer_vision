import cv2
import numpy as np

img = cv2.imread('dave.jpg') #image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
threshold = 150 #threshold
theta = np.arange(0, 180, 1) #quantization for theta
cos = np.cos(np.deg2rad(theta))
sin = np.sin(np.deg2rad(theta))
edgepoints = np.where(edges == 255)
points = list(zip(edgepoints[0], edgepoints[1]))
rho_range = edges.shape[0] + edges.shape[1]
accumulator = np.zeros((rho_range, len(theta)))
cv2.imwrite('edi.png',edges)
edimage = cv2.imread('edi.png')
for p in range(len(points)):
    for t in range(len(theta)):
        rho = int(round(points[p][1] * cos[t] + points[p][0] * sin[t]))
        accumulator[rho, t] =accumulator[rho, t]+ 1

maxmuim = np.where(accumulator > threshold)
xy = list(zip(maxmuim[0], maxmuim[1]))
for i in range(0, len(xy)):
    a = np.cos(np.deg2rad(xy[i][1]))
    b = np.sin(np.deg2rad(xy[i][1]))
    x0 = a * xy[i][0]
    y0 = b * xy[i][0]
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.line(edimage, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow('Black white image', img)
cv2.imshow('Black ', edges)
cv2.imshow('title ', edimage)


cv2.waitKey(0)
cv2.destroyAllWindows()
