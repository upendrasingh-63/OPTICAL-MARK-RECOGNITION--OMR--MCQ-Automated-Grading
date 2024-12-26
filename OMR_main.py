import numpy as np
import cv2
import utils

ans=[1,2,0,1,4]
# webCamRead=True
# cameraNo=0
# cap=cv2.VideoCapture(cameraNo)
# cap.set(10,150)

# while True:
#     if webCamRead:success,img=cap.read()
# else: img=cv2.imread(img)   

path="image.png"
img=cv2.imread(path)

img=cv2.resize(img,(600,800))
imgCountour=img.copy()
imgBiggestCountour=img.copy()
imgFinal=img.copy()

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv2.Canny(imgBlur,10,50) # detecting edges

#FINDING ALL COUNTORS
countours,hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # finding contours of the image
cv2.drawContours(imgCountour,countours,-1,(255,0,0),3) # drawing contours on the image


#FIND RECTANGES
rectCons=utils.rectContour(countours)
biggestContour=utils.getCornerPoints(rectCons[0])
graddingPoints=utils.getCornerPoints(rectCons[1])
# print(biggestContour)

if biggestContour.size!=0 and graddingPoints.size!=0:
    cv2.drawContours(imgBiggestCountour,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestCountour,graddingPoints,-1,(0,0,255),20)
    
    biggestContour= utils.reOrder(biggestContour)
    graddingPoints= utils.reOrder(graddingPoints)
    
    # defining the initial and final points for transformation
    pt1=np.float32(biggestContour)
    pt2=np.float32([[0,0],[600,0],[0,800],[600,800]])
    matrix=cv2.getPerspectiveTransform(pt1,pt2)
    #apply warp or bird perspective transformation
    imgWarpColored=cv2.warpPerspective(img,matrix,(600,800))
    
    ptG1=np.float32(graddingPoints)
    ptG2=np.float32([[0,0],[325,0],[0,150],[325,150]])
    matrixG=cv2.getPerspectiveTransform(ptG1,ptG2)
    #apply warp or bird perspective transformation
    imgGWarpColored=cv2.warpPerspective(img,matrixG,(325,150))
    # cv2.imshow("Gradding",imgGWarpColored)
    
    #apply threshold
    imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgThresh=cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow("Thresh",imgThresh)
    
    box=utils.splitBoxes(imgThresh)
    # cv2.imshow("image",box[2])
    # print(cv2.countNonZero(box[1]),cv2.countNonZero(box[2]))
    
    
    #GETTING NON ZERO PIXEL VALUE OF EACH BOX
    myPixelValu=np.zeros((5,5))
    countC=0
    countR=0
    for image in box:
        totalPixels=cv2.countNonZero(image)
        myPixelValu[countR][countC]=totalPixels
        countC+=1
        if countC==5:
            countR+=1
            countC=0
    # print(myPixelValu)
    
    #FINDING INDEX OF THE MARKERS
    myIndex=[]
    for x in range(0,5):
        arr=myPixelValu[x]
        # print("arr",arr)
        myIndexVal=np.where(arr==np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    # print(myIndex)
    
    
    #GRADING
    grading=[]
    for x in range(0,5):
        if ans[x]==myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    # print(grading)
    
    #FINAL SCORE
    score=(sum(grading)/5)*100
    # print(score,"%")
    
    
    #DISPLAYING ANSWERS
    imgResult=imgWarpColored.copy()
    imgResult=utils.showAnswers(imgResult,myIndex,grading,ans)
    imRawDrawing=np.zeros_like(imgWarpColored)
    imRawDrawing=utils.showAnswers(imRawDrawing,myIndex,grading,ans)
    InverseMatrix=cv2.getPerspectiveTransform(pt2,pt1)
    imgInverseWrap=cv2.warpPerspective(imRawDrawing,InverseMatrix,(600,800))
    
    imRawGrad=np.zeros_like(imgGWarpColored)
    cv2.putText(imRawGrad,str(int(score))+"%",(10,100),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
    # cv2.imshow("grad",imRawGrad)
    inverseMatrixG=cv2.getPerspectiveTransform(ptG2,ptG1)
    imgGInverse=cv2.warpPerspective(imRawGrad,inverseMatrixG,(600,800))#size correction, take total size to overlay on image
    # cv2.imshow("final percentage",imgGInverse)
    
    imgFinal=cv2.addWeighted(imgFinal,1,imgInverseWrap,1,0)
    imgFinal=cv2.addWeighted(imgFinal,1,imgGInverse,1,0)
    
imgBlack=np.zeros_like(img)
imgArray=([img,imgGray,imgBlur,imgCanny],
          [imgCountour,imgBiggestCountour,imgWarpGray,imgThresh],
          [imgResult,imRawGrad,imgInverseWrap,imgFinal]) 

labels=[["Original","Gray","Blur","Canny"],
        ["Contours","Biggest Con","Wrap","Thraeshold"],
        ["Result","Raw Drawing","Inv warp","Final"]]
# stackedImages=utils.stack_images(imgArray,0.5,labels)

cv2.imshow("Final Result",imgFinal)
# cv2.imshow("Original",stackedImages)
cv2.waitKey(0) # waits until a key is pressed
# if cv2.waitKey(1) & 0xff==ord('s'):
#     cv2.imwrite("output.png",imgFinal)
#     cv2.waitKey(300)