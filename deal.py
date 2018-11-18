#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from obj import *
from Funs import *
from read_xy import *

#################################################
#从图像中用cv获取目标区域
def get_objs(image):
	minVal = 100
	maxVal = 200
	MinLength = 100
	MaxLength = 800
	hwThread = 5
	
	img = image.copy()
	
	blur = cv2.GaussianBlur(img,(5,5),0)

	#Canny
	gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)	
	edges = cv2.Canny(gray,minVal,maxVal)
	
	
	#find contours
	cnts,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	#筛选轮廓
	if hierarchy is not None:
		cnts = filter_length(cnts,MinLength,MaxLength)	#根据长度过滤
		#cnts = filter_area(cnts,areaThread)
		cnts = filter_HW(cnts,hwThread)			#长宽比过滤

		objs = []
		i = 0		
		for cnt in cnts:
			x,y,w,h = cv2.boundingRect(cnt)
			box = cv2.minAreaRect(cnt)
			ob = obj([x,y,w,h],image[y:y+h,x:x+w],box)
			objs.append(ob)
			i = i + 1
	return objs

########################################
def draw_obj(obj,img):				#框处目标
	box = obj.box[:]
	#print(box)
	
	box[0] = tuple(box[0])
	box[1] = tuple(box[1])
	box = tuple(box)
	
	box = cv2.boxPoints(box)
	box = np.int0(box)
	cv2.drawContours(img,[box],0,(0,0,255),2)
	
	x,y,w,h = obj.rect
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img,str(obj.id),(x,y),font,1,(0,0,255),2)
	'''
	x,y,w,h = obj.rect
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	'''
	return img	
##################################################
#		以下代码与特征匹配相关
##################################################
def get_angle(x1,y1,x2,y2):
	if (x2-x1)!=0:
		if (x2>x1)and(y2>=y1):
			angle = np.arctan((y2-y1)/(x2-x1))
		elif (x2<x1)and(y2>=y1):
			angle = np.pi + np.arctan((y2-y1)/(x2-x1))
		elif (x2>x1)and(y2<=y1):
			angle = np.arctan((y2-y1)/(x2-x1))
		elif (x2<x1)and(y2<=y1):
			angle = -np.pi + np.arctan((y2-y1)/(x2-x1))
	else:
		if (y2>y1):
			angle = np.pi/2.0
		else:
			angle = -np.pi/2.0
	angle = angle/np.pi*180
	return angle
##################################################
#获取两个关键点的夹角
def get_lineAngle(p1,p2):
	x1,y1=p1.pt
	x2,y2=p2.pt
	angle = get_angle(x1,y1,x2,y2)
	return angle

##################################################
def find_dst(keyPoints):
	num = len(keyPoints)
	i = 0 
	dist = np.zeros([num,num])
	
	for i in range(num):
		x1,y1 = keyPoints[i].pt

		for j in range(num):
			x2,y2 = keyPoints[j].pt
			if i == j:
				dist[i][j]=0
			else:
				dist[i][j]=dist[j][i]=float(np.sqrt(np.square(x2-x1)+np.square(y2-y1)))
	return dist

##################################################
def find_maxDistPoints(keyPoints):#找最大距离点
	num = len(keyPoints)
	i = 0 
	dist = np.zeros([num,num])
	
	dist = find_dst(keyPoints)
	max_dist = 0
	dst_p1 = dst_p2 = 0
	for i in range(num):
		for j in range(num):
			if i!=j:
				if dist[i][j]>max_dist:
					dst_p1 = i
					dst_p2 = j
					max_dist = dist[i][j]
	
	if dst_p1 > dst_p2:
		t = dst_p1 	
		dst_p1 = dst_p2
		dst_p2 = t
	return dst_p1,dst_p2
#################################################
def find_destMatches(matches):
	num = len(matches)
	min_distance = 99999
	dst_match = None
	for i in  range(num):
		if matches[i].distance < min_distance:
			min_distance = matches[i].distance
			dst_match = matches[i]
	if dst_match is None:
		dst_natch = matches[0]
	return dst_match
##################################################
def imgGetAngle(test,temp,id_num):
	img1 = test.copy()
	img2 = temp.copy()
	
	img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	# Initiate SIFT detector
	sift = cv2 .xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)


	# BFMatcher with default params
	bf = cv2.BFMatcher()

	good = []
	good_matches = []
	kp1_good = []
	kp2_good = []
	
	matches = bf.knnMatch(des1,des2, k=2)
	for m,n in matches:
		if m.distance < 0.65*n.distance:
			good.append([m])
			good_matches.append(m)
			kp1_good.append(kp1[m.queryIdx])
			kp2_good.append(kp2[m.trainIdx])
	
	if len(good)<2:
		#print('no enough points')
		return None,None,None,None,None,None,None
	#根据最远匹配点计算角度
	p1,p2 = find_maxDistPoints(kp2_good)
	angle1 = get_lineAngle(kp1_good[p1],kp1_good[p2])
	angle2 = get_lineAngle(kp2_good[p1],kp2_good[p2])
	
	angle = angle2 - angle1

	if angle > 180:
		angle = angle - 360
	elif angle < -180:
		angle = angle + 360

	####################################################
	#	calculate the center point position

	x2_0,y2_0,w,h = read_xy(int(id_num))

	x1_1,y1_1 = kp1_good[p1].pt
	x1_2,y1_2 = kp1_good[p2].pt
	x2_1,y2_1 = kp2_good[p1].pt
	x2_2,y2_2 = kp2_good[p2].pt
	dist1 = np.sqrt(np.square(x1_1-x1_2)+np.square(y1_1-y1_2))
	dist2 = np.sqrt(np.square(x2_1-x2_2)+np.square(y2_1-y2_2))
	p = dist1/dist2
	
	#计算实际包洛框的w，h
	w = p*w
	h = p*h
	
	if p<1:
		'''
		match = find_destMatches(good_matches)
		x1_1,y1_1 = kp1[match.queryIdx].pt
		x2_1,y2_1 = kp2[match.trainIdx].pt
		'''
		#d = np.sqrt( np.square(x2_1-x2_0) + np.square(y2_1-y2_0) )
		a = (x2_1 - x2_0)*p
		b = (y2_1 - y2_0)*p
		p2_0 = np.array([[a],[b]])

		radian = -angle/180.0*np.pi
		H = np.array([ [np.cos(radian),-np.sin(radian)] , [np.sin(radian),np.cos(radian)] ])

		p1_0 = -np.dot(H,p2_0) + np.array([[x1_1],[y1_1]])

		x1_0,y1_0 = p1_0
			
		print str(int(id_num))+':get the center'
		return angle,kp1_good,kp2_good,x1_0,y1_0,w,h
	else:
		return angle,kp1_good,kp2_good,None,None,None,None

	return angle,kp1_good,kp2_good,None,None,None,None





