# -*- coding: utf-8 -*-

import cv2
import numpy as np
from deal_change import *

id_num = 17		#id

Test_PATH = './img_test2/test_fz/t'+str(id_num)+'_1.jpg'
Temp_PATH = './Temp/temp1/'+str(id_num)+'.jpg'
img1 = cv2.imread(Test_PATH)				#test
img2 = cv2.imread(Temp_PATH)

cv2.namedWindow('temp',0)				#Temp
cv2.imshow('test',img1)
cv2.imshow('temp',img2)


while True:
	k = cv2.waitKey(5)
	if k == ord('q'):
		break


angle,kp1_good,kp2_good,x0,y0,w0,h0 = imgGetAngle(img1,img2,id_num)

if angle is not None:
	print('angle:'+str(angle))
else:
	print('angle is None')

if kp1_good is not None:
	print len(kp1_good)
else:
	print('Not get KeyPoints')

if x0 is not None and y0 is not None:
	print x0,y0
	cv2.circle(img1,(x0,y0),3,(0,255,0),-1)
	cv2.imshow('img1',img1)
	cv2.waitKey(0)
else:
	print('not get the center')

if w0 is not None and h0 is not 0:
	print 'area: '+str(w0*h0)
else:
	print 'not get the center'
