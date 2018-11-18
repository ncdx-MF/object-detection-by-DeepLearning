#!/usr/bin/env python

import cv2 as cv

cap = cv.VideoCapture(0)
i=6

while True :
	#capture frame-by-frame
	ret,frame = cap.read()
	#print frame.shape

	#gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)	#change to gray
	cv.imshow('frame',frame)
	#cv.imshow('gray',gray)

	key = cv.waitKey(1)
	
	if  key==27 or key==ord('q'):
		break
	elif key == ord('s'):
		name = './saveimage/'+str(i)+'.jpg'
		print name
		cv.imwrite(name,frame)
		i=i+1

cap.release()
cv.destroyAllWindows()
