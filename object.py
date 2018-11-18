# -*- coding: utf-8 -*-
import sys 
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time
import os
import cv2
import numpy as np

import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from Funs import *
from detect1 import *
from obj import *
import deal_change
import deal_change2
from obj import *




global image
global x0
global y0
global angle0	
global prop
image = None

class Object(QDialog):
    
    def __init__(self,*args):
	global image
	global objs
	objs = []
        super(Object,self).__init__(*args)
        loadUi('object.ui',self)
	self.cap=cv2.VideoCapture(1)
        self.end_bd.clicked.connect(self.end_ob)
        self.start.clicked.connect(self.Start)
	self.start_ob.clicked.connect(self.Start_ob)    
	self.setMouseTracking(True)
	self.end_=0
	self.bd_flag = False
	self.ncImage=QImage('./NCU.jpg')
        self.label_img.setPixmap(QPixmap.fromImage(self.ncImage).scaled(self.label_img.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
	self.label_R.setStyleSheet("color:red")
	self.label_G.setStyleSheet("color:green")
	self.label_B.setStyleSheet("color:blue")


    def display(self,objs):
	if objs == [] or objs is None:
		return
	
	num = len(objs)
	if num > 11:
		del objs[11:]
		print('num is too many: num='+str(num))
	
	if num>=1:
		self._id1.setText(	str(int(objs[0].classes))	)
		self._X1.setText(  	str(round(objs[0].real_x,1))	)
		self._Y1.setText(  	str(round(objs[0].real_y,1))	)
		self._TH1.setText(  	str(round(objs[0].angle,1))	)
		self._S1.setText(	str(round(objs[0].real_area,3))	)
		self._credit1.setText(	str(round(objs[0].credit*100,1))+'%')
		self._figure1.setText(	str(len(objs[0].kp1))		)
		self.c_dist1.setText(	str(round(objs[0].dist,2))	)
	else:
		self._id1.setText(	'0'	)
		self._X1.setText(  	'0'	)
		self._Y1.setText(  	'0'	)
		self._TH1.setText(  	'0'	)
		self._S1.setText(	'0'	)
		self._credit1.setText(	'0'	)
		self._figure1.setText(	'0'	)
		self.c_dist1.setText(	'0'	)

	if num>=2:
		self._id2.setText(	str(int(objs[1].classes))	)
		self._X2.setText(  	str(round(objs[1].real_x,1))	)
		self._Y2.setText(  	str(round(objs[1].real_y,1))	)
		self._TH2.setText(  	str(round(objs[1].angle,1))	)
		self._S2.setText(	str(round(objs[1].real_area,3)))
		self._credit2.setText(	str(round(objs[1].credit*100,1))+'%')
		self._figure2.setText(	str(len(objs[1].kp1))		)
		self.c_dist2.setText(	str(round(objs[1].dist,2))	)
	else:
		self._id2.setText(	'0'	)
		self._X2.setText(  	'0'	)
		self._Y2.setText(  	'0'	)
		self._TH2.setText(  	'0'	)
		self._S2.setText(	'0'	)
		self._credit2.setText(	'0'	)
		self._figure2.setText(	'0'	)
		self.c_dist2.setText(	'0'	)

	if num>=3:
		self._id3.setText(	str(int(objs[2].classes))	)
		self._X3.setText(  	str(round(objs[2].real_x,1))	)
		self._Y3.setText(  	str(round(objs[2].real_y,1))	)
		self._TH3.setText(  	str(round(objs[2].angle,1))	)
		self._S3.setText(	str(round(objs[2].real_area,3))	)
		self._credit3.setText(	str(round(objs[2].credit*100,1))+'%')
		self._figure3.setText(	str(len(objs[2].kp1))		)
		self.c_dist3.setText(	str(round(objs[2].dist,2))	)
	else:
		self._id3.setText(	'0'	)
		self._X3.setText(  	'0'	)
		self._Y3.setText(  	'0'	)
		self._TH3.setText(  	'0'	)
		self._S3.setText(	'0'	)
		self._credit3.setText(	'0'	)
		self._figure3.setText(	'0'	)
		self.c_dist3.setText(	'0'	)

	if num>=4:
		self._id4.setText(	str(int(objs[3].classes))	)
		self._X4.setText(  	str(round(objs[3].real_x,1))	)
		self._Y4.setText(  	str(round(objs[3].real_y,1))	)
		self._TH4.setText(  	str(round(objs[3].angle,1))	)
		self._S4.setText(	str(round(objs[3].real_area,3))	)
		self._credit4.setText(	str(round(objs[3].credit*100,1))+'%')
		self._figure4.setText(	str(len(objs[3].kp1))		)
		self.c_dist4.setText(	str(round(objs[3].dist,2))	)
	else:
		self._id4.setText(	'0'	)
		self._X4.setText(  	'0'	)
		self._Y4.setText(  	'0'	)
		self._TH4.setText(  	'0'	)
		self._S4.setText(	'0'	)
		self._credit4.setText(	'0'	)
		self._figure4.setText(	'0'	)
		self.c_dist4.setText(	'0'	)

	if num>=5:
		self._id5.setText(	str(int(objs[4].classes))	)
		self._X5.setText(  	str(round(objs[4].real_x,1))	)
		self._Y5.setText(  	str(round(objs[4].real_y,1))	)
		self._TH5.setText(  	str(round(objs[4].angle,1))	)
		self._S5.setText(	str(round(objs[4].real_area,3))	)
		self._credit5.setText(	str(round(objs[4].credit*100,1))+'%')
		self._figure5.setText(	str(len(objs[4].kp1))		)
		self.c_dist5.setText(	str(round(objs[4].dist,2))	)
	else:
		self._id5.setText(	'0'	)
		self._X5.setText(  	'0'	)
		self._Y5.setText(  	'0'	)
		self._TH5.setText(  	'0'	)
		self._S5.setText(	'0'	)
		self._credit5.setText(	'0'	)
		self._figure5.setText(	'0'	)
		self.c_dist5.setText(	'0'	)


	if num>=6:
		self._id6.setText(	str(int(objs[5].classes))	)
		self._X6.setText(  	str(round(objs[5].real_x,1))	)
		self._Y6.setText(  	str(round(objs[5].real_y,1))	)
		self._TH6.setText(  	str(round(objs[5].angle,1))	)
		self._S6.setText(	str(round(objs[5].real_area,3))	)
		self._credit6.setText(	str(round(objs[5].credit*100,1))+'%')
		self._figure6.setText(	str(len(objs[5].kp1))		)
		self.c_dist6.setText(	str(round(objs[5].dist,2))	)
	else:
		self._id6.setText(	'0'	)
		self._X6.setText(  	'0'	)
		self._Y6.setText(  	'0'	)
		self._TH6.setText(  	'0'	)
		self._S6.setText(	'0'	)
		self._credit6.setText(	'0'	)
		self._figure6.setText(	'0'	)
		self.c_dist6.setText(	'0'	)


	if num>=7:
		self._id7.setText(	str(int(objs[6].classes))	)
		self._X7.setText(  	str(round(objs[6].real_x,1))	)
		self._Y7.setText(  	str(round(objs[6].real_y,1))	)
		self._TH7.setText(  	str(round(objs[6].angle,1))	)
		self._S7.setText(	str(round(objs[6].real_area,3))	)
		self._credit7.setText(	str(round(objs[6].credit*100,1))+'%')
		self._figure7.setText(	str(len(objs[6].kp1))		)
		self.c_dist7.setText(	str(round(objs[6].dist,2))	)
	else:
		self._id7.setText(	'0'	)
		self._X7.setText(  	'0'	)
		self._Y7.setText(  	'0'	)
		self._TH7.setText(  	'0'	)
		self._S7.setText(	'0'	)
		self._credit7.setText(	'0'	)
		self._figure7.setText(	'0'	)
		self.c_dist7.setText(	'0'	)


	if num>=8:		
		self._id8.setText(	str(int(objs[7].classes))	)
		self._X8.setText(  	str(round(objs[7].real_x,1))	)
		self._Y8.setText(  	str(round(objs[7].real_y,1))	)
		self._TH8.setText(  	str(round(objs[7].angle,1))	)
		self._S8.setText(	str(round(objs[7].real_area,3))	)
		self._credit8.setText(	str(round(objs[7].credit*100,1))+'%')
		self._figure8.setText(	str(len(objs[7].kp1))		)
		self.c_dist8.setText(	str(round(objs[7].dist,2))	)
	else:
		self._id8.setText(	'0'	)
		self._X8.setText(  	'0'	)
		self._Y8.setText(  	'0'	)
		self._TH8.setText(  	'0'	)
		self._S8.setText(	'0'	)
		self._credit8.setText(	'0'	)
		self._figure8.setText(	'0'	)
		self.c_dist8.setText(	'0'	)

	if num>=9:
		self._id9.setText(	str(int(objs[8].classes))	)
		self._X9.setText(  	str(round(objs[8].real_x,1))	)
		self._Y9.setText(  	str(round(objs[8].real_y,1))	)
		self._TH9.setText(  	str(round(objs[8].angle,1))	)
		self._S9.setText(	str(round(objs[8].real_area,3))	)
		self._credit9.setText(	str(round(objs[8].credit*100,1))+'%')
		self._figure9.setText(	str(len(objs[8].kp1))		)
		self.c_dist9.setText(	str(round(objs[8].dist,2))	)
	else:
		self._id9.setText(	'0'	)
		self._X9.setText(  	'0'	)
		self._Y9.setText(  	'0'	)
		self._TH9.setText(  	'0'	)
		self._S9.setText(	'0'	)
		self._credit9.setText(	'0'	)
		self._figure9.setText(	'0'	)
		self.c_dist9.setText(	'0'	)


	if num>=10:
		self._id10.setText(	str(int(objs[9].classes))	)
		self._X10.setText(  	str(round(objs[9].real_x,1))	)
		self._Y10.setText(  	str(round(objs[9].real_y,1))	)
		self._TH10.setText(  	str(round(objs[9].angle,1))	)
		self._S10.setText(	str(round(objs[9].real_area,3))	)
		self._credit10.setText(	str(round(objs[9].credit*100,1))+'%')
		self._figure10.setText(	str(len(objs[9].kp1))		)
		self.c_dist10.setText(	str(round(objs[9].dist,2))	)
	else:
		self._id10.setText(	'0'	)
		self._X10.setText(  	'0'	)
		self._Y10.setText(  	'0'	)
		self._TH10.setText(  	'0'	)
		self._S10.setText(	'0'	)
		self._credit10.setText(	'0'	)
		self._figure10.setText(	'0'	)
		self.c_dist10.setText(	'0'	)


	if num>=11:
		self._id11.setText(	str(int(objs[10].classes))	)
		self._X11.setText(  	str(round(objs[10].real_x,1))	)
		self._Y11.setText(  	str(round(objs[10].real_y,1))	)
		self._TH11.setText(  	str(round(objs[10].angle,1))	)
		self._S11.setText(	str(round(objs[10].real_area,3)))
		self._credit11.setText(	str(round(objs[10].credit*100,1))+'%')
		self._figure11.setText(	str(len(objs[10].kp1))		)
		self.c_dist11.setText(	str(round(objs[10].dist,2))	)
	else:
		self._id11.setText(	'0'	)
		self._X11.setText(  	'0'	)
		self._Y11.setText(  	'0'	)
		self._TH11.setText(  	'0'	)
		self._S11.setText(	'0'	)
		self._credit11.setText(	'0'	)
		self._figure11.setText(	'0'	)
		self.c_dist11.setText(	'0'	)
	
	self._num.setText(str(num))

		
	return


    def Start(self):
	global image
	print('start')
	cv2.namedWindow('img',0)
	while True:
		ret,image = self.cap.read()
		self.showImg(image)
		cv2.waitKey(5)
		if self.end_==1:
			self.end_=0
			break

    def Start_ob(self):
	global image
	global objs
	
	if not self.bd_flag:
		print('not bounding!!!')
		return

	global prop
	explor_size = 20
	print('start_ob')
	i=1
	while True:
		#################################  Clear camera cache	#######################
		self.cap.open(1)
		for m in range(30):
			ret,image = self.cap.read()
			cv2.waitKey(5)
		self.cap.release()
		
		if image is None:
			print('not get image')
			continue

		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		cv2.imwrite('./saveimage/1.jpg',image)
		
		###############################################################################
		image = Image.open('./saveimage/1.jpg')
		image = load_image_into_numpy_array(image)
		
		img = image.copy()
		
		im_width, im_height,channel = image.shape

		boxes,scores,classes,num_detections = detectImg(image)		
		
		num_box = boxes.shape[1]
		im_height,im_width,channel = image.shape
	        num_box = boxes.shape[1]
		objs = []
		#找到目标
		id_num = 0
		ex = 5	#区域扩大
		################################
		new_id_num = []
		deal1 = [4,5,13,14,17]#采用deal1计算的id
		################################

	        for i in range(30):
			ymin, xmin, ymax, xmax = boxes[0][i]
			(left, right, top, bottom) = (	xmin * im_width - ex,
							xmax * im_width + ex,
							ymin * im_height - ex, 
							ymax * im_height + ex)
		  	left, right, top, bottom = np.int0([left, right, top, bottom])
		  	if scores[0][i]>0.5:
			
				if classes[0][i] == 18:
					continue
				#######################		Filter out duplicate ID
				if classes[0][i] in new_id_num:
					continue
				new_id_num.append(classes[0][i])
				#########################################################
				if image[top:bottom,left:right] == []:
					continue
				ob = obj(	id_num,
						[left,top,right-left,bottom-top],
						image[top:bottom,left:right],
						classes[0][i],
						scores[0][i] )
				

				id_num = id_num+1
				

				temp = cv2.imread('./Temp/temp4/'+str(int(ob.classes))+'.jpg')	#更具类型读取模板图


				#不同类别采用了不同方式获得旋转角
				if ob.classes in deal1:
					angle,kp1,kp2,x0,y0,w,h = deal_change.imgGetAngle1(ob.img,temp,ob.classes)	#特征匹配获取旋转角
				else:
					angle,kp1,kp2,x0,y0,w,h = deal_change2.imgGetAngle2(ob.img,temp,ob.classes)	#特征匹配获取旋转角


				ob.setDist(1.00)				
				if angle is None:
					print('id '+str(ob.id)+": Don't get Angle!!!!!")
					ob.setMatches([],[])
					ob.setDist(1.00)
				else:
					#已经根据特征点获取到了中心坐标以及旋转角
					ob.angle = angle
					ob.setMatches(kp1,kp2)	#将匹配到的点储存起来
					print('id'+str(ob.id)+": get angle "+str(angle))
					#获取到中心
					if x0 is not None and y0 is not None:
						cv2.circle(img,(int(left+x0),int(top+y0)),3,(0,255,0),-1)
						dist = distance(left+x0,top+y0,ob.pix_x,ob.pix_y)
						
						if dist < 6.0/prop:	#计算出的中心贴近boundingbox中心，则储存计算中心，保留角度等计算信息
							if dist < 0.35*6.0/prop:
								ob.setCenter(left+x0,top+y0)
							ob.setArea(w,h)
							ob.setDist(dist/(6.0/prop))
						else:
							ob.setDist(1.00)
					else:
						ob.setDist(1.00)
							
			
				objs.append(ob)
				#画出目标
				color = tuple(map(int,150*np.random.rand(3)))			
				cv2.rectangle(img,(left,top),(right,bottom),color,2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(img,str(ob.id),(left,top),font,1,color,2)
				cv2.circle(img,(int((right+left)/2.0),int((bottom+top)/2.0)),3,(0,0,255),-1)


		self.display(objs)
		self.showImg(img)
		
		cv2.waitKey(5000)
	
	

    def showImg(self,img):
	if img is not None:
		height, width, bytesPerComponent = img.shape
		bytesPerLine = bytesPerComponent * width
		img2=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.image=QImage(img2.data,width,height,bytesPerLine,QImage.Format_RGB888)
		self.camera.setPixmap(QPixmap.fromImage(self.image).scaled(self.camera.width(),self.camera.height()))
		self.camera.setPixmap(QPixmap.fromImage(self.image))


    def end_ob(self):
	global image
	global x0
	global y0
	global angle0	
	global prop

	'''
	ground_Long = float(self.wl_long.text())
	ground_wide = float(self.wl_wide.text())
	lt_x = float(self.pix_x1.text())
	lt_y = float(self.pix_y1.text())
	lb_x = float(self.pix_x2.text())
	lb_y = float(self.pix_y2.text())
	rt_x = float(self.pix_x3.text())
	rt_y = float(self.pix_y3.text())
	rb_x = float(self.pix_x4.text())
	rb_y = float(self.pix_y4.text())
	'''

	ground_Long = float(50)
	ground_wide = float(70)
	lt_x = float(10)
	lt_y = float(10)
	lb_x = float(10)
	lb_y = float(80)
	rt_x = float(100)
	rt_y = float(10)
	rb_x = float(100)
	rb_y = float(80)

	w = np.sqrt(np.square(rt_x - lt_x) + np.square(rt_y - lt_y))
	h = np.sqrt(np.square(lb_x - lt_x) + np.square(lb_y - lt_y))
	p = (ground_Long + ground_wide)/(w+h)
	
	x0 = lt_x
	y0 = lt_y
	angle0 = deal_change2.get_angle(lt_x,lt_y,rt_x,rt_y)
	prop = p

	set_globalVars(x0,y0,angle0,prop)
	get_globalVars()
	
	self.label_angle0.setText(	str(round(angle0,3))	)
	self.label_prop.setText(	str(round(p,3))		)
	
	self.bd_flag = True


    def mouseMoveEvent(self,event):
        global image
        pointX =event.x()
        pointY =event.y() 
        if 40<pointX and pointX<640+40 and pointY>50 and pointY<480+50 and image is not None:
           pointX = pointX - 40
           pointY = pointY - 50
           x = int(pointX)
           y = int(pointY)

           B,G,R = image[y,x,:]          

           self.label_R.setText("R:%s" % R)
	   self.label_G.setText("G:%s" % G)
	   self.label_B.setText("B:%s" % B)
           self.label_groudx.setText("%s"%(pointX))
           self.label_groudy.setText("%s"%(pointY))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',"Are you sure to quit?", QMessageBox.Yes |QMessageBox.No, QMessageBox.No)
 
        if reply == QMessageBox.Yes:
            event.accept()
            self.cap.release()
	    exit()
        else:
            event.ignore() 



if __name__=='__main__':
    app=QApplication(sys.argv)
    objec=Object()
    if objec.exec_():
      objec.loginEnvent()
    else:
       print("quit")
       sys.exit(1)
    sys.exit(app.exec_())
