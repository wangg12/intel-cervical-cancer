#!/usr/bin/python

###############################################################################
# Name		: ObjectMarker.py
# Author	: Python implementation: sqshemet 
# 	 	  Original ObjectMarker.cpp: http://www.cs.utah.edu/~turcsans/DUC_files/HaarTraining/
# Date		: 7/24/12
# Description	: Object marker utility to be used with OpenCV Haar training. 
#		  Tested on Ubuntu Linux 10.04 with OpenCV 2.1.0.
# Usage		: python ObjectMarker.py outputfile (full path)inputdirectory
#
# Additions by Paul: Multiple rectangles, possibility to delete last rectangle
# Ongoing	: Trying to keep track of what is left to do in the folder (save and come back later)
# WARNINGS	: When using <a>, if no rectangle is added, the last rectangle drawn will be saved twice
###############################################################################

import cv2
from cv2 import cv
import sys
import os
import glob
import platform

IMG_SIZE = (720,1280)
IMG_CHAN = 3
IMG_DEPTH = cv.IPL_DEPTH_8U
image = cv.CreateImage(IMG_SIZE, IMG_DEPTH, IMG_CHAN)
image2 = cv.CreateImage(IMG_SIZE, IMG_DEPTH, IMG_CHAN) 
roi_x0 = 0
roi_y0 = 0
roi_x1 = 0
roi_y1 = 0
num_of_rec = 0
start_draw = False
window_name = "<Space> to save and load next, <X> to skip, <ESC> to exit, <a> to add rectangle, <d> to delete last rectangle"

def on_mouse(event, x, y, flag, params):
	global start_draw
	global roi_x0
	global roi_y0
	global roi_x1
	global roi_y1
	global image2

	if (event == cv.CV_EVENT_LBUTTONDOWN):
		print("LButton")
		if (not start_draw):
			roi_x0 = x
			roi_y0 = y
			start_draw = True
		else:
			roi_x1 = x
			roi_y1 = y
			start_draw = False


	elif (event == cv.CV_EVENT_MOUSEMOVE and start_draw):
		#Redraw ROI selection
		image2 = cv.CloneImage(image)
		if(len(rect_list)>0):
			for coord in rect_list:
				cv.Rectangle(image2,coord[0],coord[1],
					cv.CV_RGB(255,0,0),5
				)
		cv.Rectangle(image2, (roi_x0, roi_y0), (x,y), 
			cv.CV_RGB(255,0,255),5)
		cv.ShowImage(window_name, image2)
	#cv2.resizeWindow(window_name, int(round(width/2)),int(round(height/2)))



def main():
# Might want to divide this a bit more.
	global image
	global rect_list
	global height
	global width
	iKey = 0
	if (len(sys.argv) != 3):
		sys.stderr.write("%s output_info.txt ALB\n" 
			% sys.argv[0])
		return -1

	input_directory = sys.argv[2]
	output_file = sys.argv[1]

	#Get a file listing of all files within the input directory
	try:
		if(platform.system()=='Linux'):
			files = glob.glob(input_directory+"/*.jpg")
		else:
			files = glob.glob(input_directory+"\*.jpg")

	except OSError:
		sys.stderr.write("Failed to open directory %s\n" 
			% input_directory)
		return -1

	files.sort()

	# init GUI
	cv.NamedWindow(window_name, cv2.WINDOW_NORMAL)
	cv.SetMouseCallback(window_name, on_mouse, None)

	sys.stderr.write("Opening directory...")
	# init output of rectangles to the info file
	os.chdir(input_directory)
	sys.stderr.write("done.\n")

	str_prefix = input_directory
	
	if(os.path.isfile(output_file)):
		if(os.path.isfile("tmp.txt")):
			print("restarting from saved tmp")
			with open("tmp.txt") as file:
    				lines = []
    				for line in file:
    					lines.append(line.rstrip())
			files = lines
			
		else:
			print("creating tmp")
			tmp = open("tmp.txt",'w')
			for file in files:
				tmp.write("%s\n" % os.path.basename(file))
			tmp.close()
		output = open(output_file, 'a')
	else:
		try:
			output = open(output_file, 'w')
			print("creating tmp")
			tmp = open("tmp.txt",'w')
			for file in files:
				tmp.write("%s\n" % file)
			tmp.close()
		except IOError:
			sys.stderr.write("Failed to open file %s.\n" % output_file)
			return -1

	for file in files:
		coordinates=""
		str_postfix = ""
		num_of_rec = 0
		if(platform.system()=='Linux'):
			img = str_prefix + file
		else:
			img = file
		sys.stderr.write("Loading image %s...\n" % img)

		try: 
			image = cv.LoadImage(img)
		except IOError: 
			sys.stderr.write("Failed to load image %s.\n" % img)
			return -1

		#  Work on current image
		cv.ShowImage(window_name, image)
		width, height = cv.GetSize(image)
		cv2.resizeWindow(window_name, int(round(width/2)),int(round(height/2)))
		####  Changed
		# Need to figure out waitkey returns.
		# <ESC> = 43		exit program
		# <Space> = 48		add rectangle to current image
		# <x> = 136		skip image
		
		#iKey = cv.WaitKey(0) % 255
		# This is ugly, but is actually a simplification of the C++.
		### Paul update:
		#### Unix
		# <ESC> = 1048603	exit program
		# <Space> = 1048608	add rectangle to current image
		# <x> = 1048696		skip image
		# <a> = 1048673 	add another rectangle
		# <d> = 1048676		delete last rectangle
		### Windows
		# <ESC> = 27	exit program
		# <Space> = 32	add rectangle to current image
		# <x> = 120		skip image
		# <a> = 97 	add another rectangle
		# <d> = 100		delete last rectangle
		### Keylist
		if(platform.system()=='Linux'):
			key_esc=1048603
			key_space=1048608
			key_x=1048696
			key_a=1048673
			key_d=1048676
		else:
			key_esc=27
			key_space=32
			key_x=120
			key_a=97
			key_d=100

		
		alreadySaved=False
		NotExit=True
		rect_list=[]
		
		while NotExit:
			iKey = cv.WaitKey(0)
			

			if iKey == key_esc:
				cv.DestroyWindow(window_name)
				return 0
				NotExit
			elif iKey == key_a: ## Add
				sys.stderr.write("Adding rectangle")
				num_of_rec = num_of_rec + 1
					### rectangle is: min x, min y, abs(x1-x0), abs(y1-y0)
					#if (roi_x0<roi_x1 and roi_y0<roi_y1):
					#	str_postfix += " %d %d %d %d" % (roi_x0,
					#		roi_y0, (roi_x1-roi_x0), (roi_y1-roi_y0))
					#elif (roi_x0>roi_x1 and roi_y0>roi_y1):
					#	str_postfix += " %d %d %d %d" % (roi_x1, 
					#		roi_y1, (roi_x0-roi_x1), (roi_y0-roi_y1))
					#elif (roi_x0>roi_x1 and roi_y0<roi_y1): ### Added draw possibilities but not sure # Paul
					#	str_postfix += " %d %d %d %d" % (roi_x1, 
					#		roi_y0, (roi_x0-roi_x1), (roi_y1-roi_y0))
					#else:
					#	str_postfix += " %d %d %d %d" % (roi_x0,
					#		roi_y1, (roi_x1-roi_x0), (roi_y0-roi_y1))
				str_postfix += " %d %d %d %d" %(min(roi_x0,roi_x1), min(roi_y0,roi_y1),
						abs(roi_x0-roi_x1),abs(roi_y0-roi_y1) )
				coordinates = str_postfix
				sys.stderr.write("Coordinates: %s \n" % coordinates)
				sys.stderr.write("Number of rectangles (global): %s \n" % num_of_rec)

				rect_list.append([(min(roi_x0,roi_x1),min(roi_y0,roi_y1)),
						(max(roi_x0,roi_x1),max(roi_y0,roi_y1))])
				print(rect_list)
					#sys.stderr.write("Coordistr_postfix: %s \n" % str_postfix)
			elif iKey == key_d: ## delete last rectangle
				if (num_of_rec>0):
					num_of_rec = num_of_rec - 1
					splitted=coordinates.split(" ")
					del splitted[-4:]
					coordinates=" ".join(splitted)
					sys.stderr.write("Coordinates: %s \n" % coordinates)
					del rect_list[-1]
				sys.stderr.write("Number of rectangles (global): %s \n" % num_of_rec)				
			elif iKey == key_space:
				num_of_rec += 1
				sys.stderr.write("Number of rectangles (global): %s \n" % num_of_rec)
				str_postfix = " %d %d %d %d" %(min(roi_x0,roi_x1), min(roi_y0,roi_y1),
						abs(roi_x0-roi_x1),abs(roi_y0-roi_y1) )
				output.write(img + " " + str(num_of_rec) + coordinates + str_postfix +"\n")
				NotExit=False
				# remove entry from tmp
				with open(input_directory+"/tmp.txt", 'r') as fin:
    					data = fin.read().splitlines(True)
				with open(input_directory+"/tmp.txt", 'w') as fout:
    					fout.writelines(data[1:])

			elif iKey == key_x:
				sys.stderr.write("Skipped %s.\n" % img)
				NotExit=False
				# remove entry from tmp
				with open(input_directory+"/tmp.txt", 'r') as fin:
    					data = fin.read().splitlines(True)
				with open(input_directory+"/tmp.txt", 'w') as fout:
    					fout.writelines(data[1:])
		
if __name__ == '__main__':
	main()
