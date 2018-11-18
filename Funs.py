#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from obj import *

#######################################
def nothing(x):
	pass
#######################################
def carry(num,n=2):
	num = num*np.power(10,n)
	num = num + 0.5
	num = int(num)
	num = float(num)*np.power(0.1,n)
	return num
#######################################
def filter_length(cnts,MinLength,MaxLength=10000):	#长度筛选
	new_cnts = []
	for cnt in cnts:	
		length = cv2.arcLength(cnt,False)
		if length >= MinLength and length <= MaxLength:
			new_cnts.append(cnt)
	return new_cnts
############################################
def filter_HW(cnts,thread):				#长宽比筛选
	new_cnts = []
	for cnt in cnts:
		x,y,w,h = cv2.boundingRect(cnt)
		if w>h:
			t=w
			w=h
			h=t
	
		if h/float(w)<=thread:
			new_cnts.append(cnt)		
	return new_cnts
##############################################
def filter_area(cnts,thread):				#面积筛选
	new_cnts = []
	for cnt in cnts:
		x,y,w,h =cv2.boundingRect(cnt)
		area = w*h
		if area > thread:
			new_cnts.append(cnt)
	return new_cnts
#################################################
def filter_matchShapes(cnts,sample_cnts,ret_Threshold):	#轮廓匹配
	new_cnts=[]
	for cnt in cnts:
		min_ret = 10000
		for sample_cnt in sample_cnts:
			ret = cv2.matchShapes(cnt[0],sample_cnt[0],1,0.0)
			if ret <= min_ret:
				min_ret = ret
		if min_ret<=ret_Threshold:
			new_cnts.append(cnt)
	return new_cnts			
###################################################
def draw_box(img,cnts):					#根据轮廓画box
	areas = []
	i = 0
	for cnt in cnts:	
		#get rectangle	
		x,y,w,h = cv2.boundingRect(cnt)
		#areas.append([x,y,w,h])
		#get min_rectangle
		min_rect = cv2.minAreaRect(cnt)
		#print 'angle:'+str(min_rect[2])
		box = cv2.boxPoints(min_rect)
		box = np.int0(box)	
		
		frame = cv2.drawContours(img,[box],-1,(0,0,255),2)
		frame = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		
		font=cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,str(i),(x,y), font, 1,(0,0,255),2)

		#cv2.imshow('box'+str(i),img[y:y+h,x:x+w])
		i=i+1
		
	return	areas
################################################
def calibration(imgs):					#标定
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	imgpoints=[]
	for img in imgs:
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
		#print corners.shape
		if ret == True:
			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			
			imgpoints.append(corners2)
			# Draw and display the corners
			img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
			cv2.imshow('p',img)
		
	return
####################################################
def find_corner(im):					#找角点
	img = im[:]
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# find Harris corners
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	dst = cv2.dilate(dst,None)
	
	ret,dst = cv2.threshold(dst,0.1*dst.max(),255,0)

	coners = np.argwhere(dst == 255)		

	minlength = 1000000
	maxlength = 0
	pointTL = []
	pointBR = []
	for coner in coners:
		length = coner[0]*coner[0]+coner[1]*coner[1]
		if length < minlength:
			minlength = length
			pointTL = coner
		if length > maxlength:
			maxlength = length
			pointBR = coner
	t = pointTL[0]	
	pointTL[0] = pointTL[1]	
	pointTL[1] = t
	t = pointBR[0]	
	pointBR[0] = pointBR[1]	
	pointBR[1] = t	
		
	return pointTL,pointBR

###############################################
def find_corner2(im):					#找标定框
	img = im.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,100,200)
	#cv2.imshow('edges',edges)
	ret,cnts,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
	cnts = filter_length(cnts,800)		
	max_area = 0
	for cnt in cnts:
		x,y,w,h = cv2.boundingRect(cnt)
		area = w*h
		if area > max_area:
			max_area = area
			dst_cnt = cnt
	
	x,y,w,h = cv2.boundingRect(dst_cnt)
	#box = cv2.minAreaRect(dst_cnt)
	
	#return dst_cnt,[x,y],[x+w,y+h],box
	img = img[y:y+h,x:x+w]
	return img
		
##################################################
def distance(x0,y0,x1,y1):
	distance = np.sqrt(np.square(x0-x1)+np.square(y0-y1))
	return distance




