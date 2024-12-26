import cv2
import numpy as np

# make function to stack images which take input arrays of image, scale and list of labels which is optional and this function will return stacked image
def stack_images(img_array, scale, labels=[]):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2: 
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2: 
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    
    if len(labels) != 0:
        each_img_width = int(ver.shape[1] / cols)
        each_img_height = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * each_img_width, each_img_height * d), 
                              (c * each_img_width + len(labels[d][c]) * 13 + 27, 30 + each_img_height * d), 
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels[d][c], (each_img_width * c + 10, each_img_height * d + 20), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

# def rectContour(contour):
#     rect=cv2.minAreaRect(contour)
#     box=cv2.boxPoints(rect)
#     box=np.int0(box)
#     return box

def rectContour(contour):
    rectCon=[]
    for i in contour:
        area=cv2.contourArea(i)
        # print("Area", area)
        if area>50:
            pari=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*pari,True)
            # print("countour",len(approx))
            if len(approx)==4:
                rectCon.append(i)
    rectCon=sorted(rectCon,key=cv2.contourArea,reverse=True)
    # print(len(rectCon))
    return rectCon

def getCornerPoints(cont):
    peri=cv2.arcLength(cont,True)
    approx=cv2.approxPolyDP(cont,0.02*peri,True)
    return approx


def reOrder(myPoint):
    myPoint = myPoint.reshape((4, 2))
    myNewPoint = np.zeros((4, 1, 2), np.int32)
    add = myPoint.sum(1)
    # print(myPoint)
    # print(add)
    myNewPoint[0] = myPoint[np.argmin(add)]
    myNewPoint[3] = myPoint[np.argmax(add)]
    diff = np.diff(myPoint, axis=1)
    myNewPoint[1] = myPoint[np.argmin(diff)]
    myNewPoint[2] = myPoint[np.argmax(diff)]
    # print(myNewPoint)
    # print(diff)
    return myNewPoint

def splitBoxes(img):
    rows=np.vsplit(img,5)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
            # cv2.imshow("Split",box)
    return boxes

def showAnswers(img,myIndex,grading,ans):
    secW=img.shape[1]//5
    secH=img.shape[0]//5
    for x in range(0,5):
        myAns=myIndex[x]
        cX=(myAns*secW)+secW//2
        cY=(x*secH)+secH//2
        if grading[x]==1:
            myColor=(0,255,0)
        else:
            myColor=(0,0,255)
            correctAns=ans[x]
            cv2.circle(img,(correctAns*secW+secW//2,x*secH+secH//2),30,(0,255,0),cv2.FILLED)
        cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
    return img