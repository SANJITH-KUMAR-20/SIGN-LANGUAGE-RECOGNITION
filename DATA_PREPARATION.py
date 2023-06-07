import cv2 as cv
import numpy as np


#Let us declare dummy variables
background = None #we create a background variable so that we can later use it to have a background relative to our foreground(hand) to compute the acc_weight
accum_weight = 0.5

# Then we initialize the co-ordinates for the area of intrest(AOI)
AOI_top = 100
AOI_bottom = 300+50
AOI_right =150+50
AOI_left = 350+50
def segmented_img(frame,threshold=25):
    global background

    diff = cv.absdiff(background.astype('uint8'),frame)
    ret , thresh = cv.threshold(diff,threshold,255,cv.THRESH_BINARY)
    contours, hierar = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if len(contours)== 0:
        return None
    else:
        hand = max(contours,key=cv.contourArea)
        return(thresh,hand)


def calculate_acc_avg(frame,accum_weight):
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    cv.accumulateWeighted(frame,background,accum_weight)

cam = cv.VideoCapture(0)# 0 refers to the built in camera,if you have other cameras installed you may use 1,2.. respectively

num_of_frames = 0
num_of_imgs = 0
element = 10

while True:
    ret,frame = cam.read()# reads in a single frame every instant returns a boolean value and the frame
    frame = cv.flip(frame,1) 
    '''Inverts a frame to'''

    copy_frame = frame.copy()

    aoi = frame[AOI_top:AOI_bottom,AOI_right:AOI_left]
    gray = cv.cvtColor(aoi,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(9,9),0)

    if num_of_frames < 60:
        calculate_acc_avg(gray,accum_weight)
        if num_of_frames<=59:
            cv.putText(copy_frame,"FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),1)
    elif num_of_frames<=300:
        img = segmented_img(gray)
        cv.putText(copy_frame,"Adjust hand...Gesture for" + str(element), (200, 400), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
        if img is not None:
            thresh, hand_seg = img
            cv.drawContours(copy_frame,[hand_seg + (AOI_right,AOI_top)],-1,(0,0,255),1  )

            cv.putText(copy_frame,str(num_of_frames)+"For"+str(element),(70,45),cv.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),1)

            cv.imshow("Threshholded hand image",thresh)
    else:
            Img = segmented_img(gray)

            if img is not None:
                thresh,hand_seg = Img
                
                cv.drawContours(copy_frame,[hand_seg+(AOI_right,AOI_top)],-1,(0,0,255),1)
                
                cv.putText(copy_frame,str(num_of_frames),(70,45),cv.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),1)
                cv.putText(copy_frame, str(num_of_imgs)+"images"+"For"+str(element),(200,400),cv.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),1)

                cv.imshow("Threshholded hand image",thresh)
                if num_of_imgs<=300:

                      cv.imwrite(r"C:\\Users\\sanji\\Desktop\\SIGN LANGUAGE RECOGNITION\\DATA_FOR_TRANS\\TRAIN\\3\\"+str(num_of_imgs)+".jpg",thresh)
                else:
                    break
                num_of_imgs+=1
            else:
                cv.putText(copy_frame, 'No hand detected...', (200, 400), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
    cv.rectangle(copy_frame, (AOI_left, AOI_top), (AOI_right, AOI_bottom), (255,128,0), 3)
    num_of_frames+=1
    cv.imshow("Sign Detection",copy_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break



cv.destroyAllWindows()
cam.release()
