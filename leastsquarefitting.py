import cv2
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

img = cv2.imread('L.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
hzimg =cv2.flip(edges, 0)
imgtest =cv2.flip(img, 0)
edgepoints = np.where(hzimg == 255)
points = list(zip(edgepoints[1], edgepoints[0]))
"""
least square matrix form
B =(XTX-1)X-1*Y 

"""
print(len(edgepoints[1]))

x1=edgepoints[1] #x axis
y1=edgepoints[0] #y axis
help=np.ones((len(edgepoints[1]),), dtype=int)

print(x1.shape)
x1 = np.vstack((help, x1)) #modify x axis matrix
print(x1)

xt=np.transpose(x1) #x transpose

inverse = np.linalg.inv((np.matmul(x1,xt))) # inverse of x*xt


t = np.matmul(xt,inverse)
print(t.shape)
print(y1.shape)
B = np.matmul(y1,t)

yline=[]
for i in edgepoints[1]:
    yline.append(B[0]+B[1]*i) #calculate y values using B0 and B1 using line equation
print(yline)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.scatter(edgepoints[1],edgepoints[0],  color='black')
plt.plot(edgepoints[1], yline, color='blue', linewidth=3)
#edimage =hzimg.copy()
cv2.imwrite('edi.png',hzimg)
edimage = cv2.imread('edi.png')

for i in range(0, len(yline)):


    start = (math.floor(edgepoints[1][i]), math.floor(yline[i]))

    image = cv2.line(imgtest, start, start, (0, 0, 255), 2) #drawing line on photo
    edimage2= cv2.line(edimage, start, start, (0, 0, 255), 2)

limage =cv2.flip(image, 0)
ledimage =cv2.flip(edimage2, 0)



cv2.imshow('Black white image', limage)
cv2.imshow('Black  image', edges)
cv2.imshow('Black', ledimage)


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
