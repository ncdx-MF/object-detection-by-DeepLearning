

def read_xy(id):
	f = open('./Temp/temp4/data.txt')
	line = f.readlines()
	x = float(line[(id-1)*6+1].rstrip())
	y = float(line[(id-1)*6+2].rstrip())
	w = float(line[(id-1)*6+3].rstrip())
	h = float(line[(id-1)*6+4].rstrip())
	f.close()
	return x,y,w,h
