import numpy as np
import cv2
from collections import deque
import HandTrackingModule as htm



# default called trackbar function

def setValues(x):
    print('')


# Giving different arrays to handle colour
# points of different colour These arrays
# will hold the points of a particular colour
# in the array which will further be used
# to draw on canvas

tipIds = [4, 8, 12, 16, 20]

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
vpoints = [deque(maxlen=1024)]

# These indexes will be used to mark position
# of pointers in colour array

black_index = 0
green_index = 0
red_index = 0
voilet_index = 0

# The kernel to be used for dilation purpose

kernel = np.ones((5, 5), np.uint8)

# The colours which will be used as ink for
# the drawing purpose

# colors = [(0, 0, 0), (254,5,85), (23,154,0), (0, 40, 255)]
colors = [(0, 0, 0), (255,0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0

# Here is code for Canvas setup

paintWindow = np.zeros((471, 636, 3)) + 0xFF

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Loading the default webcam of PC.

cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.75)

# Keep looping

while True:

    # Reading the frame from the camera

    (ret, frame) = cap.read(0)
    frame = cv2.flip(frame, 1)
    img = detector.findHands(frame)
    lmList = detector.findPosition(img, draw=False)

    fingers = []

    if(len(lmList)!=0):
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(lmList[4])
        totalFingers = fingers.count(1)


    # Flipping the frame to see same side of yours

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Getting the updated positions of the trackbar
    # and setting the HSV values


    frame = cv2.circle(frame,(40,90), 20, (255,255,255),-1)
    frame = cv2.circle(frame,(40,140), 20, (0,0,0),-1)
    frame = cv2.circle(frame,(40,190),20,(255,0,0),-1)
    frame = cv2.circle(frame,(40,240), 20, (0,255,0),-1)
    frame = cv2.circle(frame,(40,290), 20, (0,0,255),-1)

    cv2.putText(
        frame,
        'C',
        (32,94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,0,0),
        2,
        cv2.LINE_AA,
    )

    center = None
    # Ifthe contours are formed

    if len(lmList) != 0 and totalFingers==1:
        # Get the radius of the enclosing circle
        # around the found contour
        #x = 640 - lmList[8][1]
        #y = lmList[8][2]
        lst = lmList[tipIds[fingers.index(1)]]
        x,y = lst[1],lst[2]
        #print(x,y)
        # Draw the circle around the contour

        cv2.circle(frame, (x,y), int(20), (0, 0xFF, 0xFF), 2)

        # Calculating the center of the detected contour
        center = (x, y)

        # Now checking if the user wants to click on
        # any button above the screen

        if center[0] <= 60:

            # Clear Button

            if 70 <= center[1] <= 110:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                vpoints = [deque(maxlen=512)]

                black_index = 0
                green_index = 0
                red_index = 0
                voilet_index = 0

                paintWindow[67:, :, :] = 0xFF
            elif 120 <= center[1] <= 160:
                colorIndex = 0  # Black
            elif 170 <= center[1] <= 210:
                colorIndex = 1  # Voilet
            elif 220 <= center[1] <= 260:
                colorIndex = 2  # Green
            elif 270 <= center[1] <= 310:
                colorIndex = 3  # Red
        else:
            if colorIndex == 0:
                bpoints[black_index].appendleft(center)
            elif colorIndex == 1:
                vpoints[voilet_index].appendleft(center)
            elif colorIndex == 2:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 3:
                rpoints[red_index].appendleft(center)
    else:

    # Append the next deques when nothing is
    # detected to avois messing up

        bpoints.append(deque(maxlen=512))
        black_index += 1
        vpoints.append(deque(maxlen=512))
        voilet_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        

    # Draw lines of all the colors on the
    # canvas and frame

    points = [bpoints, vpoints, gpoints, rpoints]
    for i in range(len(points)):

        for j in range(len(points[i])):

            for k in range(1, len(points[i][j])):

                if points[i][j][k - 1] is None or points[i][j][k] \
                    is None:
                    continue

                cv2.line(frame, points[i][j][k - 1], points[i][j][k],
                         colors[i], 6)
                cv2.line(paintWindow, points[i][j][k - 1],
                         points[i][j][k], colors[i], 6)

    # Show all the windows

    cv2.imshow('Tracking', frame)
    cv2.imshow('Paint', paintWindow)
    # cv2.imshow('mask', Mask)
    # If the 'q' key is pressed then stop the application

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and all resources

cap.release()
cv2.destroyAllWindows()
