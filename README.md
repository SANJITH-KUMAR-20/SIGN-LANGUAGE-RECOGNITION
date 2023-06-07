# SIGN-LANGUAGE-RECOGNITION
This repository contains the code for a simple sign language recognizer using OpenCV and Tensorflow
1. DATA_PREPARATION.py
2. Train_model.ipynb
3. GENERATEOUTPUT.py

--->DATA_PREPARATION.py:
    This file contains the code for collecting data using the live camera feed
    1. Create a set of folders to properly store the data according to categories
            1. Create 'Train' and 'Test' folders having 10 folders each starting from 1 to 10
            2. This is to store the collected data in different folders
    2. Once the camera starts a metric called "Accumulated Weighted Average" is calculated using cv.accumulateweighted()
            1. You can learn more about this metric here - https://www.geeksforgeeks.org/background-subtraction-in-an-image-using-concept-of-running-average/
            2. Once the accumulated weighted average is calculated from 60 frames, we simply subtract it from a frame or image containing an object(stationary or moving) which will result in the extraction of the foreground of the image or frame.
            3. We find the threshold for the foreground and save it to the proper folder and find the contours.
            4. Then we train the model

--->Train_model.ipynb:
      1. This file contains the CNN model, we train the model with the collected data from the previous step.
      2. We save the model as .h5 file.
      
--->GENERATEOUTPUT.py:
      1. We load in the downloaded model and take live feed from the camera and use the thresholded foreground as input to the model to predict the result and us cv.putText to print it on the frame. 
 
