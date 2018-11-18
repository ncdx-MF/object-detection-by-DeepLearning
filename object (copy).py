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
from deal import *
from obj import *
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
from detect import *
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
	self.cap=cv2.VideoCapture(0)
        self.end_bd.clicked.connect(self.end_ob)
        self.start.clicked.connect(self.Start)
	self.start_ob.clicked.connect(self.Start_ob)    
	self.setMouseTracking(True)
	self.end_=0
	self.bd_flag = False



    def display(self,objs):
	if objs == [] or objs is None:
		return
	
	num = len(objs)
	if num > 11:
		del objs[11:]
		print('num is too many: num='+str(num))
	
	if num>=1:
		self._id1.setText(	str(objs[0].classes)	)
		self._X1.setText(  	str(objs[0].real_x)	)
		self._Y1.setText(  	str(objs[0].real_y)	)
		self._TH1.setText(  	str(objs[0].angle)	)
		self._S1.setText(	str(objs[0].real_area)	)
	else:
		self._id1.setText(	'0'	)
		self._X1.setText(  	'0'	)
		self._Y1.setText(  	'0'	)
		self._TH1.setText(  	'0'	)
		self._S1.setText(	'0'	)

	if num>=2:
		self._id2.setText(	str(objs[1].classes)	)
		self._X2.setText(  	str(objs[1].real_x)	)
		self._Y2.setText(  	str(objs[1].real_y)	)
		self._TH2.setText(  	str(objs[1].angle)	)
		self._S2.setText(	str(objs[1].real_area)	)
	else:
		self._id2.setText(	'0'	)
		self._X2.setText(  	'0'	)
		self._Y2.setText(  	'0'	)
		self._TH2.setText(  	'0'	)
		self._S2.setText(	'0'	)

	if num>=3:
		self._id3.setText(	str(objs[2].classes)	)
		self._X3.setText(  	str(objs[2].real_x)	)
		self._Y3.setText(  	str(objs[2].real_y)	)
		self._TH3.setText(  	str(objs[2].angle)	)
		self._S3.setText(	str(objs[2].real_area)	)
	else:
		self._id3.setText(	'0'	)
		self._X3.setText(  	'0'	)
		self._Y3.setText(  	'0'	)
		self._TH3.setText(  	'0'	)
		self._S3.setText(	'0'	)

	if num>=4:
		self._id4.setText(	str(objs[3].classes)	)
		self._X4.setText(  	str(objs[3].real_x)	)
		self._Y4.setText(  	str(objs[3].real_y)	)
		self._TH4.setText(  	str(objs[3].angle)	)
		self._S4.setText(	str(objs[3].real_area)	)
	else:
		self._id4.setText(	'0'	)
		self._X4.setText(  	'0'	)
		self._Y4.setText(  	'0'	)
		self._TH4.setText(  	'0'	)
		self._S4.setText(	'0'	)


	if num>=5:
		self._id5.setText(	str(objs[4].classes)	)
		self._X5.setText(  	str(objs[4].real_x)	)
		self._Y5.setText(  	str(objs[4].real_y)	)
		self._TH5.setText(  	str(objs[4].angle)	)
		self._S5.setText(	str(objs[4].real_area)	)
	else:
		self._id5.setText(	'0'	)
		self._X5.setText(  	'0'	)
		self._Y5.setText(  	'0'	)
		self._TH5.setText(  	'0'	)
		self._S5.setText(	'0'	)


	if num>=6:
		self._id6.setText(	str(objs[5].classes)	)
		self._X6.setText(  	str(objs[5].real_x)	)
		self._Y6.setText(  	str(objs[5].real_y)	)
		self._TH6.setText(  	str(objs[5].angle)	)
		self._S6.setText(	str(objs[5].real_area)	)
	else:
		self._id6.setText(	'0'	)
		self._X6.setText(  	'0'	)
		self._Y6.setText(  	'0'	)
		self._TH6.setText(  	'0'	)
		self._S6.setText(	'0'	)


	if num>=7:
		self._id7.setText(	str(objs[6].classes)	)
		self._X7.setText(  	str(objs[6].real_x)	)
		self._Y7.setText(  	str(objs[6].real_y)	)
		self._TH7.setText(  	str(objs[6].angle)	)
		self._S7.setText(	str(objs[6].real_area)	)
	else:
		self._id7.setText(	'0'	)
		self._X7.setText(  	'0'	)
		self._Y7.setText(  	'0'	)
		self._TH7.setText(  	'0'	)
		self._S7.setText(	'0'	)


	if num>=8:		
		self._id8.setText(	str(objs[7].classes)	)
		self._X8.setText(  	str(objs[7].real_x)	)
		self._Y8.setText(  	str(objs[7].real_y)	)
		self._TH8.setText(  	str(objs[7].angle)	)
		self._S8.setText(	str(objs[7].real_area)	)
	else:
		self._id8.setText(	'0'	)
		self._X8.setText(  	'0'	)
		self._Y8.setText(  	'0'	)
		self._TH8.setText(  	'0'	)
		self._S8.setText(	'0'	)


	if num>=9:
		self._id9.setText(	str(objs[8].classes)	)
		self._X9.setText(  	str(objs[8].real_x)	)
		self._Y9.setText(  	str(objs[8].real_y)	)
		self._TH9.setText(  	str(objs[8].angle)	)
		self._S9.setText(	str(objs[8].real_area)	)
	else:
		self._id9.setText(	'0'	)
		self._X9.setText(  	'0'	)
		self._Y9.setText(  	'0'	)
		self._TH9.setText(  	'0'	)
		self._S9.setText(	'0'	)


	if num>=10:
		self._id10.setText(	str(objs[9].classes)	)
		self._X10.setText(  	str(objs[9].real_x)	)
		self._Y10.setText(  	str(objs[9].real_y)	)
		self._TH10.setText(  	str(objs[9].angle)	)
		self._S10.setText(	str(objs[9].real_area)	)
	else:
		self._id10.setText(	'0'	)
		self._X10.setText(  	'0'	)
		self._Y10.setText(  	'0'	)
		self._TH10.setText(  	'0'	)
		self._S10.setText(	'0'	)


	if num>=11:
		self._id11.setText(	str(objs[10].classes)	)
		self._X11.setText(  	str(objs[10].real_x)	)
		self._Y11.setText(  	str(objs[10].real_y)	)
		self._TH11.setText(  	str(objs[10].angle)	)
		self._S11.setText(	str(objs[10].real_area)	)
	else:
		self._id11.setText(	'0'	)
		self._X11.setText(  	'0'	)
		self._Y11.setText(  	'0'	)
		self._TH11.setText(  	'0'	)
		self._S11.setText(	'0'	)

		
	return


    def Start(self):
	global image
	print('start')
	while True:
		ret,image = self.cap.read()
		
		cv2.imshow('img',image)
		self.showImg(image)
		#cv2.destroyAllWindows()
		#print image.shape
		cv2.waitKey(5)
		if self.end_==1:
			self.end_=0
			break

    def Start_ob(self):
	global image
	global objs
	'''
	if not self.bd_flag:
		print('not bounding!!!')
		return
	'''
	explor_size = 20
	print('start_ob')
	i=1
	while True:
		ret,image = self.cap.read()
		
		im_width, im_height,channel = image.shape
		boxes,scores,classes,num_detections = detect(image)		
		
		num_box = boxes.shape[1]
		im_height,im_width,channel = image.shape
	        num_box = boxes.shape[1]
		objs = []
		#找到目标
		id_num = 0
		ex = 5 			#区域扩大
	        for i in range(30):
			ymin, xmin, ymax, xmax = boxes[0][i]
			(left, right, top, bottom) = (	xmin * im_width - ex,
							xmax * im_width + ex,
							ymin * im_height - ex, 
							ymax * im_height + ex)
		  	left, right, top, bottom = np.int0([left, right, top, bottom])
		  	if scores[0][i]>0.5:
				#print(scores[0][1])
				#print(boxes[0][i])
				if classes[0][i] == 13:
					continue
		   		cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
				ob = obj(	id_num,
						[left,top,right-left,bottom-top],
						image[top:bottom,left:right],
						classes[0][i],
						scores[0][i] )			
				id_num = id_num+1

				temp = cv2.imread('./Temp/'+str(int(ob.classes))+'.jpg')#更具类型读取模板图
				#print(temp.shape)
				#print(ob.img.shape)
				angle = imgGetAngle(temp,ob.img)#特征匹配获取旋转角
				if angle is not None:
					ob.angle = angle
					print('id '+str(ob.id)+":  Don't get Angle")
				print(ob.angle)
				
				objs.append(ob)
		for ob in objs:
			image = draw_obj(ob,image)
		self.display(objs)
		self.showImg(image)
		
		cv2.waitKey(1)
	
	

    def showImg(self,img):
	
	if img is not None:
		height, width, bytesPerComponent = img.shape
		bytesPerLine = bytesPerComponent * width
		img2=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.image=QImage(img2.data,width,height,bytesPerLine,QImage.Format_RGB888)
		self.camera.setPixmap(QPixmap.fromImage(self.image).scaled(self.camera.width(),self.camera.height()))


    def end_ob(self):
	global image
	global x0
	global y0
	global angle0	
	global prop
	'''
	ground_Long = float(self.wl_long.toPlainText())
	ground_wide = float(self.wl_wide.toPlainText())
	lt_x = float(self.pix_x1.text())/500.0*640.0
	lt_y = float(self.pix_y1.text())/330.0*480.0
	lb_x = float(self.pix_x2.text())/500.0*640.0
	lb_y = float(self.pix_y2.text())/330.0*480.0
	rt_x = float(self.pix_x3.text())/500.0*640.0
	rt_y = float(self.pix_y3.text())/330.0*480.0
	rb_x = float(self.pix_x4.text())/500.0*640.0
	rb_y = float(self.pix_y4.text())/330.0*480.0
	lt_x = 88/500.0*640.0
	lt_y = 82/330.0*480.0
	lb_x = 80/500.0*640.0
	lb_y = 285/330.0*480.0	
	w = rt_x - lt_x
	h = lb_y - lt_y
	p = (ground_Long + ground_wide)/(w+h)
 	#set_globalVars(lt_x,lt_y,0,p)
	set_globalVars(lt_x,lt_y,0,p)
	get_globalVars()
	'''
	self.bd_flag = True


    def mouseMoveEvent(self,event):
        global image
        pointX =event.x()
        pointY =event.y() 
        if 50<pointX and pointX<550 and pointY>20 and pointY<350 and image is not None:
           pointX = pointX - 50
           pointY = pointY - 20
           x = int(pointX/500.0*480.0)
           y = int(pointY/330.0*640.0)
           #print x,y
           B,G,R = image[x,y,:]          
           #print B,G,R
           self.label_c.setText("(%s,%s,%s)" % (R,G,B))
           self.label_groudx.setText("%s"%(pointX))
           self.label_groudy.setText("%s"%(pointY))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',"Are you sure to quit?", QMessageBox.Yes |QMessageBox.No, QMessageBox.No)
 
        if reply == QMessageBox.Yes:
            event.accept()
            self.cap.release()
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
