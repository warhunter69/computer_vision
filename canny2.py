import cv2
import numpy as np
import math

class EdgeDetection(object):
    """
    def __init__(self, image, GSimg, Gaus, Fder, Sder,Sobil,Prewitt):
        self.image = image
        self.GSimg = GSimg
        self.Gaus = Gaus
    """


    def firstDerivativeEdgeDetector(self,img):
        xkernel = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
        ykernel = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

        ix = self.convolute(img,xkernel) # convultion with dervitavie mask in x axis
        iy = self.convolute(img,ykernel)  # convultion with dervitavie mask in y axis

        return ix,iy
    def sobil(self,img):
        xkernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ykernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        ix = self.convolute(img, xkernel)  # convultion with dervitavie mask in x axis
        iy = self.convolute(img, ykernel)  # convultion with dervitavie mask in y axis

        return ix,iy
    def secondDerivativeEdgeDetector(self,img):
        xkernel = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
        ykernel = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
        ix = self.convolute(img, xkernel)  # convultion with dervitavie mask in x axis
        iy = self.convolute(img, ykernel)  # convultion with dervitavie mask in y axis
        return ix, iy



    def Prewitt (self,img):
        xkernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        ykernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        ix = self.convolute(img, xkernel)  # convultion with dervitavie mask in x axis
        iy = self.convolute(img, ykernel)  # convultion with dervitavie mask in y axis

        return ix,iy

    def gaussianMask(self,gray,size,sigma):
        r = math.floor(size / 2)
        # change padding size
        image = np.pad(gray, ((r, r), (r, r)), 'constant')
        output = np.zeros_like(gray)
        kernel = np.zeros((size, size))
        # r=math.floor(size/2)
        x1 = math.floor(size / 2)
        x2 = -x1
        y1 = x1
        y2 = -x1
        while y2 != y1 + 1:
            x2 = -r
            while x2 != x1 + 1:
                print(x2, y2)
                z = 2 * np.pi * (sigma ** 2)
                w = np.exp(-(x2 ** 2 + y2 ** 2) / (2 * sigma ** 2))
                kernel[y2 + r, x2 + r] = (1 / z) * w
                x2 = x2 + 1

            y2 = y2 + 1
        #print(kernel)
        #print(kernel.sum())
        #print(kernel * (1 / kernel.sum()))
        kernel = kernel * (1 / kernel.sum())
        for x in range(gray.shape[1]):
            for y in range(gray.shape[0]):
                # change shift size
                matrix = image[y: y + size, x: x + size]
                output[y, x] = (kernel * matrix).sum()
        return output

    def non_maximasuppression(self,GM,Grad):
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


    def convolute(self,img,kernel):
        image = np.pad(img, ((1, 1), (1, 1)), 'constant')
        

        output = np.zeros_like(img)
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):

                matrix = image[y: y + 3, x: x + 3]
                output[y, x] = (kernel * matrix).sum()
                t= np.sum(np.multiply(kernel, matrix))
                output[y,x]=abs(t)
                #if(abs(output[y,x])<127):
                    #output[y, x]=0
                #else:
                    #output[y, x]=255

        return output
    def double_threshold(self,NMS,lowThresholdRatio,highThresholdRatio):


        val = np.copy(NMS)
        h = int(val.shape[0])
        w = int(val.shape[1])
        highThreshold = np.max(val) * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        x = 0.1
        oldx = 0

        while (oldx != x):
            oldx = x
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if (val[i, j] > highThreshold):
                        val[i, j] = 1
                    elif (val[i, j] < lowThreshold):
                        val[i, j] = 0
                    else:
                        if ((val[i - 1, j - 1] > highThreshold) or
                                (val[i - 1, j] > highThreshold) or
                                (val[i - 1, j + 1] > highThreshold) or
                                (val[i, j - 1] > highThreshold) or
                                (val[i, j + 1] > highThreshold) or
                                (val[i + 1, j - 1] > highThreshold) or
                                (val[i + 1, j] > highThreshold) or
                                (val[i + 1, j + 1] > highThreshold)):
                            val[i, j] = 1
            x = np.sum(val == 1)

        val = (val == 1) * val

        return val


    def Canny(self,gray):
        size=3
        sigma=1
        Sgray = self.gaussianMask(gray,size,sigma)
        cv2.imshow('sg', Sgray)

        ix, iy = self.sobil(Sgray)
        GM = np.zeros(shape=(ix.shape))
        Dir = np.zeros(shape=(ix.shape))
        for i in range(0, len(ix[0])):
            for j in range(0, len(ix)):
                # print(i,j)
                xn = pow(ix[j][i], 2)
                yn = pow(iy[j][i], 2)
                GM[j][i] = pow(xn + yn, 1 / 2)  # gradiant magnitude

                Dir[j][i] = math.degrees(math.atan2(iy[j][i], ix[j][i]))  # gradiant direction
                if (Dir[j][i] < 0):
                    Dir[j][i] = Dir[j][i] + 360

        NMS = self.non_maximasuppression(GM,Dir)
        cv2.imshow('nms', NMS)
        lowThresholdRatio = 0.01
        highThresholdRatio = 0.1


        DT = self.double_threshold(NMS,lowThresholdRatio,highThresholdRatio)

        return DT
def main():

    img = cv2.imread('dave.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    obj = EdgeDetection()
    size = 3
    sigma = 1
    Sgray = obj.gaussianMask(gray,size,sigma)
    xsobil,ysobil=obj.sobil(Sgray)
    xpre,ypre = obj.Prewitt(Sgray)
    xfr, yfr = obj.firstDerivativeEdgeDetector(Sgray)
    xsec,ysec = obj.secondDerivativeEdgeDetector(Sgray)
    canny = obj.Canny(gray)
    cv2.imshow('xfr', xfr)
    cv2.imshow('yfr', yfr)
    cv2.imshow('xsec', xsec)
    cv2.imshow('ysec', ysec)
    cv2.imshow('xsobil', xsobil)
    cv2.imshow('ysobil', ysobil)
    cv2.imshow('xprew', xpre)
    cv2.imshow('ypre', ypre)
    cv2.imshow('canny', canny)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
