import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False):
  print("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('/Users/anubhavjain/Desktop/sign-language-mnist/outpy.mp4',cv2.VideoWriter_fourcc('M','P','4','2'), 10, (frame_width,frame_height))

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    out.write(frame)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.rectangle(frame, (150, 150), (500, 500), (0, 255, 0), 0)
    crop_image = frame[150:486, 150:486]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret1, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find contour with maximum area
    contour = max(contours, key=lambda x: cv2.contourArea(x))

    # Create bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

    hand = crop_image[y:y+h,x:x+w]
    # Display the resulting frame

    cv2.imshow('Gesture',mat=frame)

    if cv2.waitKey(25) & 0xFF == ord('c'):
        cv2.imwrite('/Users/anubhavjain/Desktop/sign-language-mnist/outpy.png',crop_image)
        img = cv2.imread('/Users/anubhavjain/Desktop/sign-language-mnist/outpy.png',1)
        img1 = cv2.imread('/Users/anubhavjain/Desktop/sign-language-mnist/outpy.png',0)
        img1 = cv2.GaussianBlur(img1,(5,5),0)
        img1 = cv2.Canny(img1,140,220)

        cv2.imwrite('/Users/anubhavjain/Desktop/sign-language-mnist/outpy2.png',img1)
        r,c,x = img.shape

        pic_arrayr = np.empty((r,c))
        pic_arrayg = np.empty((r, c))
        pic_arrayb = np.empty((r, c))
        for i in range(r):
            for j in range(c):
                pic_arrayr[i][j] = img.item(i,j,0)/255

        for i in range(r):
            for j in range(c):
                pic_arrayg[i][j] = img.item(i,j,1)/255

        for i in range(r):
            for j in range(c):
                pic_arrayb[i][j] = img.item(i,j,2)/255

        print(pic_arrayr,pic_arrayr.shape)
        print(pic_arrayg, pic_arrayg.shape)
        print(pic_arrayb, pic_arrayb.shape)

        img =   cv2.resize(img,None,fx=1/12,fy=1/12,interpolation=cv2.INTER_AREA)

        cv2.imwrite('/Users/anubhavjain/Desktop/sign-language-mnist/outpy1.png', img)
        img = cv2.imread('/Users/anubhavjain/Desktop/sign-language-mnist/outpy1.png',0)
        pic_array = np.empty((28, 28))
        for i in range(28):
            for j in range(28):
                pic_array[i][j] = float(img.item(i,j))
                pic_array[i][j] = pic_array[i][j]/255

        #pic_array = pic_array.reshape(1, 28, 28, 1)
        df = pd.DataFrame(pic_array)
        df.to_csv('/Users/anubhavjain/Desktop/sign-language-mnist/outpy3.csv')

        df1 = pd.read_csv('/Users/anubhavjain/Desktop/sign-language-mnist/sign_mnist_train.csv')
        x_train = np.array(df1)
        print(x_train[0])

        print(pic_array,pic_array.shape)
        print(df.loc[0, 0])

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break;

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()