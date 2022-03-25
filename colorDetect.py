

import numpy as np
import cv2
import copy
from jproperties import Properties
configs = Properties()


COLOR_ROWS = 80
COLOR_COLS = 250

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise RuntimeError('Error opening VideoCapture.')

(grabbed, frame) = capture.read()
snapshot = np.zeros((640, 480, 3), dtype=np.uint8)

colorArray = np.zeros((COLOR_ROWS, COLOR_COLS, 3), dtype=np.uint8)
cv2.imshow('Color', colorArray)

recordColorArray = [255,255,255]
recorded = False

def getColorXY( x, y, record):
    colorArray[:] = snapshot[y, x, :]
    rgb = snapshot[y, x, [2,1,0]]
	
	# From stackoverflow/com/questions/1855884/determine-font-color-based-on-background-color
    luminance = 1 - (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) / 255
    if luminance < 0.5:
        textColor = [0,0,0]
    else:
        textColor = [255,255,255]

    cv2.putText(colorArray, str(rgb), (20, COLOR_ROWS - 20),
		fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=textColor)
    cv2.imshow('Color', colorArray)
    global recorded
    global recordColorArray

    if 	recorded == True and compareColors(recordColorArray, colorArray[1,1,:]):
        print("recorded ", recordColorArray);
        print("colorArray", colorArray[1,1,:]);
        print("similar");

    if record == True:
        recordColorArray =  copy.deepcopy(colorArray[1,1,:])
        recorded = True
		
def compareColors(recordColorArray1 , colorArray1):
	if abs(recordColorArray1[0]-colorArray1[0]) < 12 and abs(recordColorArray1[1]-colorArray1[1]) < 12 and 	abs(recordColorArray1[2]-colorArray1[2]) < 12:
		return True
	else:
		print (" vals ", abs(recordColorArray1[0]-colorArray1[0]), abs(recordColorArray1[1]-colorArray1[1]),	abs(recordColorArray1[2]-colorArray1[2]))
		return False

while True:
    (grabbed, frame) = capture.read()
    cv2.imshow('Video', frame)
	
    if not grabbed:
        break

    keyVal = cv2.waitKey(1) & 0xFF
    if keyVal == ord('q'):
        break
    elif keyVal == ord('r'):
        getColorXY (320, 240, True)

    snapshot = frame.copy()
    getColorXY (320, 240, False)
    #width = capture.get(3)
    #height = capture.get(4)
    #fps = capture.get(5)
    #print('width, height, fps:', width, height, fps)
    

capture.release()
cv2.destroyAllWindows()