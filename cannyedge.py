import cv2
import numpy as np
import math


def NonMaxSup(GM, Grad):#nonmaximum suppression
        NMS = np.zeros(GM.shape)
        for i in range(1, int(GM.shape[0]) - 1):
            for j in range(1, int(GM.shape[1]) - 1):
                if ((Grad[i, j] >= -22.5 and Grad[i, j] <= 22.5) or (Grad[i, j] <= -157.5 and Grad[i, j] >= 157.5)):
                    if ((GM[i, j] > GM[i, j + 1]) and (GM[i, j] > GM[i, j - 1])):
                        NMS[i, j] = GM[i, j]
                    else:
                        NMS[i, j] = 0
                if ((Grad[i, j] >= 22.5 and Grad[i, j] <= 67.5) or (Grad[i, j] <= -112.5 and Grad[i, j] >= -157.5)):
                    if ((GM[i, j] > GM[i + 1, j + 1]) and (GM[i, j] > GM[i - 1, j - 1])):
                        NMS[i, j] = GM[i, j]
                    else:
                        NMS[i, j] = 0
                if ((Grad[i, j] >= 67.5 and Grad[i, j] <= 112.5) or (Grad[i, j] <= -67.5 and Grad[i, j] >= -112.5)):
                    if ((GM[i, j] > GM[i + 1, j]) and (GM[i, j] > GM[i - 1, j])):
                        NMS[i, j] = GM[i, j]
                    else:
                        NMS[i, j] = 0
                if ((Grad[i, j] >= 112.5 and Grad[i, j] <= 157.5) or (Grad[i, j] <= -22.5 and Grad[i, j] >= -67.5)):
                    if ((GM[i, j] > GM[i + 1, j - 1]) and (GM[i, j] > GM[i - 1, j + 1])):
                        NMS[i, j] = GM[i, j]
                    else:
                        NMS[i, j] = 0

        return NMS


def thresholding(img):#Hysteresis Thresholding

    lowThresholdRatio = 0
    highThresholdRatio =0.1
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    x = 0.1
    oldx = 0


    while (oldx != x):
        oldx = x
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if (GSup[i, j] > highThreshold):
                    GSup[i, j] = 1
                elif (GSup[i, j] < lowThreshold):
                    GSup[i, j] = 0
                else:
                    if ((GSup[i - 1, j - 1] > highThreshold) or
                            (GSup[i - 1, j] > highThreshold) or
                            (GSup[i - 1, j + 1] > highThreshold) or
                            (GSup[i, j - 1] > highThreshold) or
                            (GSup[i, j + 1] > highThreshold) or
                            (GSup[i + 1, j - 1] > highThreshold) or
                            (GSup[i + 1, j] > highThreshold) or
                            (GSup[i + 1, j + 1] > highThreshold)):
                        GSup[i, j] = 1
        x = np.sum(GSup == 1)

    GSup = (GSup == 1) * GSup

    return GSup


img = cv2.imread('dave.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),5) #gaussian filter

blur1 = cv2.GaussianBlur(gray,(5,5),15)
xkernel = np.array([[0 ,0,0],[1 ,0,-1],[0 ,0,0]])
ykernel =  np.array([[0 ,1,0],[0,0,0],[0 ,-1,0]])
prwkernel = np.array([[1 ,2,1],[0 ,0,0],[-1 ,-1,-1]])
xsobel = np.array([[1 ,2,1],[0 ,0,0],[-1 ,-2,-1]])
ysobel = np.array([[1 ,0,-1],[2 ,0,-2],[1 ,0,-1]])
print(len(blur[0]),len(blur))



print(xkernel)

ix= cv2.filter2D(blur,ddepth=cv2.CV_16S,kernel=xkernel) #convultion with dervitavie mask in x axis
iy = cv2.filter2D(blur,ddepth=cv2.CV_16S,kernel=ykernel)#convultion with dervitavie mask in y axis



#cv2.imshow('ix', ix)
#cv2.imshow('iy', iy)
GM = np.zeros(shape=(ix.shape))
Dir = np.zeros(shape=(ix.shape))
print("yo",math.degrees(math.atan2(-1,1))+360)

for i in range(0,len(ix[0])):
    for j in range(0,len(ix)):
        #print(i,j)
        xn = pow(ix[j][i],2)
        yn = pow(iy[j][i],2)
        GM[j][i] = pow(xn + yn,1/2) #gradiant magnitude

        Dir[j][i]=math.degrees(math.atan2(iy[j][i],ix[j][i])) #gradiant direction
        if(Dir[j][i]<0):
            Dir[j][i] = Dir[j][i] + 360

nms = NonMaxSup(GM, Dir)
output=thresholding(nms)
cv2.imshow('wwww', img)

cv2.imshow('blur', blur)

cv2.imshow('GM', nms)
cv2.imshow('output', output)



cv2.waitKey(0)
cv2.destroyAllWindows()
