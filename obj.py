#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from Funs import *

global x0
global y0
global angle0	
global prop




class obj(object):

	def __init__(self,rect,img,box):
		##############
		self.id=0	
		self.classes = []	#类别
		self.box = []		#最小包洛框  [[x,y],[w,h],angele]

		self.pix_x = []
		self.pix_y = []
		self.pix_area = []

		self.real_x = []
		self.real_y = []
		self.real_area = []
	
		self.angle = []
	
		self.img = []		#图片
		self.rect = []		#图片的包洛矩形  [x,y,w,h]
		self.credit = 0		#可信度
		self.classes = 0
		#################
		self.rect = rect
		self.img = img
		self.box = box
		self.classes = classes
		self.credit = confident
		self.calculate()
		return

	def __init__(self,id_num,rect,img,classes,confident):
		##############	
		self.id=id_num
		self.classes = []	#类别
		self.box = []		#最小包洛框  [[x,y],[w,h],angele]

		self.pix_x = []
		self.pix_y = []
		self.pix_area = []

		self.real_x = []
		self.real_y = []
		self.real_area = []

		self.angle = []	
	
		self.img = []		#图片
		self.rect = []		#图片的包洛矩形  [x,y,w,h]
		self.credit = 0		#可信度
		#################
		self.rect = rect
		self.img = img
		self.classes = classes
		self.credit = confident
		self.box = self.getBox()
		self.calculate()
		return
	
	def calculate(self):
		global x0
		global y0
		global angle0	
		global prop
		
		self.pix_x = int(self.rect[0]+self.rect[2]/2.0)
		self.pix_y = int(self.rect[1]+self.rect[3]/2.0)
		self.real_x = prop*(self.rect[0]+self.rect[2]/2.0 - x0)
		self.real_y = prop*(self.rect[1]+self.rect[3]/2.0 - y0)

		if self.box is None:
			self.angle = 0.0
			self.pix_area =  0.0
			self.real_area = 0.0
			return
		self.angle = self.box[2] - angle0
		self.pix_area =  int(self.box[1][0]*self.box[1][1])
		self.real_area = prop*prop*(self.box[1][0]*self.box[1][1])
	
		return

	def getBox(self):#获取最小框
		if self.img ==[] or self.img is None:
			return
	
		img = self.img.copy()
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray,50,200)
		ret,cnts,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
		#cnts = filter_length(cnts,120)
		max_length = 0
		dst_cnt = None
		for cnt in cnts:
			length = cv2.arcLength(cnt,False)
			if length > max_length :
				max_length = length
				dst_cnt = cnt
		if dst_cnt is None:
			return None
			
		box=cv2.minAreaRect(dst_cnt)
		box = list(box)
		box[0] = list(box[0])
		box[1] = list(box[1])
		box[0][0] = self.rect[0] + box[0][0]
		box[0][1] = self.rect[1] + box[0][1]
		return box
	def setMatches(self,kp1,kp2):
		self.kp1 = kp1
		self.kp2 = kp2
		
	def setCenter(self,x,y):
		global x0
		global y0
		global angle0	
		global prop
		self.pix_x = int(x)
		self.pix_y = int(y)
		self.real_x = prop*(x-x0)
		self.real_y = prop*(y-y0)

	def setArea(self,pix_w,pix_h):
		global prop
		self.pix_area = pix_w*pix_h
		self.real_area = prop*prop*(pix_w*pix_h)

	def setDist(self,distance):
		self.dist = distance
		
	
def set_globalVars(x,y,angle,p):
	global x0
	global y0
	global angle0	
	global prop
	x0 = x
	y0 = y
	angle0 = angle
	prop = p

def get_globalVars():
	global x0
	global y0
	global angle0	
	global prop
	print x0
	print y0
	print angle0
	print prop



