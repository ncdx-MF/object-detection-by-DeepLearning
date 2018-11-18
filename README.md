# object-detection-by-DeepLearning
Use faster RCNN and opencv to pinpoint the location, rotation Angle, and area of the object

## 参考工程地址：

https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/

## 模型文件

faster rcnn weight.tar文件 </br>
链接：https://pan.baidu.com/s/1E0dce81mVb8K6SPYZ6nGug  </br>
提取码：8t1l 

## 使用方法

python3 object.py

## 注意事项

运行前注意需要摄像头，在object.py里修改运行所需的摄像头设备 </br>

line 48 </br>
self.cap=cv2.VideoCapture(1) #这里修改使用的摄像头序号  </br>
line 317 </br>
self.cap.open(1)             #以及这里 </br> 

运行程序先按开始键启动摄像头，根据返回的画面进行标定，按标定结束键后确定标定，在按识别键开始识别
