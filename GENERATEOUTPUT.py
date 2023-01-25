import numpy as np
import cv2 as cv
import keras

word = ['ONE','TWO','THREE','FOUR','FIVE']
model = keras.models.load_model(r"Trained_model.h5")

background = None
accum_weigth = 0.5


def segmented(frame,threshs=25):
    global background
    diff = cv.absdiff(background.astype('uint8'),frame)

    ret,thresh = cv.threshold(diff,threshs,255,cv.THRESH_BINARY)

    contours, hier = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    if len(contours)==0:
        return None
    else:
        hand_max_cnt = max(contours,key=cv.contourArea)
        return(thresh,hand_max_cnt)


def calculate_accavg(frame,accum_weight):
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    cv.accumulateWeighted(frame,background,accum_weight)
AOI_top = 100
AOI_bottom = 300+50
AOI_right =150+50
AOI_left = 350+50


cam = cv.VideoCapture(0)
num_of_frames = 0
while True:
    ret, frame = cam.read()
    copy_frame = frame.copy()
    aoi = frame[AOI_top:AOI_bottom,AOI_right:AOI_left]

    gray = cv.cvtColor(aoi,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(9,9),0)

    if num_of_frames<70:
        calculate_accavg(gray,accum_weight=accum_weigth)
        
        cv.putText(copy_frame,"Fetching BackGround...",(80,400),cv.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),1)
    else:
        hand = segmented(gray)

        if hand is not None:
            thresh,segment = hand

            cv.drawContours(copy_frame,[segment+(AOI_right,AOI_top)],-1,(0,0,255),1)
            cv.imshow("Thresholded Hand",thresh)

            thresh = cv.resize(thresh,(64,64))
            thresh = cv.cvtColor(thresh,cv.COLOR_GRAY2RGB)
            thresh = np.reshape(thresh,(1,thresh.shape[0],thresh.shape[1],3))

            prediction = model.predict(thresh)
            print(prediction)
            cv.putText(copy_frame, word[np.argmax(prediction)],(170,45),cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv.rectangle(copy_frame,(AOI_left,AOI_top),(AOI_right,AOI_bottom),(255,128,0),3)
    num_of_frames+=1

    cv.imshow("DECTION",copy_frame)

    k = cv.waitKey(1) & 0xFF

    if k== 27:
        break

cam.release()
cv.destroyAllWindows()