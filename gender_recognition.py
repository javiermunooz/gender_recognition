# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import shutil
import cvlib as cv

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
	help="path to input image")
ap.add_argument("-d", "--dir",
	help="path to input directory")
ap.add_argument("-m", "--nomen", action='store_true', default="False", help="Delete men pictures")
ap.add_argument("-w", "--nowomen", action='store_true', default="False", help="Delete women pictures")
ap.add_argument("-o", "--oneperson", action='store_true', default="False", help="Delete pictures with more than one person")
ap.add_argument("-t","--nometadata", action='store_true', default="False", help="Delete all metadata files")
ap.add_argument("--hide", action='store_true', default="False", help="Hide output")
args = ap.parse_args()

# download pre-trained model file (one-time download)
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())

# read input image
image = cv2.imread(args.image)

classes = ['man','woman']

# Considering a directory
if args.dir is not None:

	# load pre-trained model
	model = load_model(model_path)

	for filename in os.listdir(args.dir):
		path=os.path.join(args.dir,filename)

		'Delete file if it is not an image'
		if (path.endswith('.json') or path.endswith('.txt')) and args.nometadata==True:
			os.remove(path)
			print(filename + " was removed")
			continue
		else:
			if path.endswith('.jpg') or path.endswith('.jpeg'):
				file = cv2.imread(path)
				if file is None:
					print("Could not read input image")
					exit()
			else:
				print("Skipping",filename)
				continue

		# detect faces in the image
		face, confidence = cv.detect_face(file)
		people=len(face)
		
		# loop through detected faces
		for idx, f in enumerate(face):

			 # get corner points of face rectangle       
			(startX, startY) = f[0], f[1]
			(endX, endY) = f[2], f[3]
			
			#fixes double-face bug
			if endX>1020 or endY>1020 or startX<0 or endY<0:
			    people=people-1
			    break

			# draw rectangle over face
			cv2.rectangle(file, (startX,startY), (endX,endY), (0,255,0), 2)

			# crop the detected face region
			face_crop = np.copy(file[startY:endY,startX:endX])

			# preprocessing for gender detection model
			try:
				face_crop = cv2.resize(face_crop, (96,96))
				face_crop = face_crop.astype("float") / 255.0
				face_crop = img_to_array(face_crop)
				face_crop = np.expand_dims(face_crop, axis=0)


				# apply gender detection on face
				conf = model.predict(face_crop)[0]
				
				# show decision in console
				#print(conf)
				#print(classes)

				# get label with max accuracy
				idx = np.argmax(conf)
				label = classes[idx]
				label_text=label
			
			except:
			   pass

			try:
				label = "{}: {:.2f}%".format(label, conf[idx] * 100)
			except:
				pass
			
			Y = startY - 10 if startY - 10 > 10 else startY + 10

			# write label and confidence above face rectangle
			cv2.putText(file, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
						0.7, (0, 255, 0), 2)
			
			# if -m option is enabled, remove men's pics			
			if label_text=="man" and args.nomen==True:
				try:
					os.remove(path)
					print(filename + " was removed")
				except:
					pass
			
			# if -w option is enabled, remove women's pics
			if label_text=="woman" and args.nowomen==True:
				try:
					os.remove(path)
					print(filename + " was removed")
				except:
					pass
		
		# if -o option is enabled, remove pictures with more than one person or none
		if people!=1 and args.oneperson==True:
			try:
				os.remove(path)
				print(filename + " was removed")
			except:
				pass
		
		# if --hide option is off, show the picture and its rectangle
		if args.hide=="False":
			# display output
			cv2.imshow("gender detection", file)

			# press any key to close window           
			cv2.waitKey()

			# save output
			cv2.imwrite("gender_detection.jpg", file)

# Considering a single image	
elif args.image is not None:
    
    file = cv2.imread(args.image)
    if file is None:
	    print("Could not read input image")
	    exit()

	# load pre-trained model
    model = load_model(model_path)

	# detect faces in the image
    face, confidence = cv.detect_face(file)
    people=len(face)

	# loop through detected faces
    for idx, f in enumerate(face):

		# get corner points of face rectangle       
	    (startX, startY) = f[0], f[1]
	    (endX, endY) = f[2], f[3]
		
	    # fixes double-face bug
	    if endX>1020 or endY>1020 or startX<0 or endY<0:
		    people=people-1
		    break

		# draw rectangle over face
	    cv2.rectangle(file, (startX,startY), (endX,endY), (0,255,0), 2)

		# crop the detected face region
	    face_crop = np.copy(file[startY:endY,startX:endX])

		# preprocessing for gender detection model
	    face_crop = cv2.resize(face_crop, (96,96))
	    face_crop = face_crop.astype("float") / 255.0
	    face_crop = img_to_array(face_crop)
	    face_crop = np.expand_dims(face_crop, axis=0)

		# apply gender detection on face
	    conf = model.predict(face_crop)[0]
						
		# show decision in console
	    print(conf)
	    print(classes)
			
		# get label with max accuracy
	    idx = np.argmax(conf)
	    label = classes[idx]
	    label_text=label
		
	    label = "{}: {:.2f}%".format(label, conf[idx] * 100)
					
	    Y = startY - 10 if startY - 10 > 10 else startY + 10

		# write label and confidence above face rectangle
	    cv2.putText(file, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 255, 0), 2)
			
		# if -m option is enabled, remove men's pics
	    if label_text=="man" and args.nomen==True:
		    os.remove(args.image)
		    print(args.image + " was removed")
		    break
		
		# if -w option is enabled, remove women's pics
	    if label_text=="woman" and args.nowomen==True:
		    os.remove(args.image)
		    print(args.image + " was removed")
		    break
	
	# if -o option is enabled, remove pictures with more than one person
    if people!=1 and args.oneperson==True:
        os.remove(args.image)
        print(args.image + " was removed")
	
	# if --hide option is off, show the picture and its rectangle
    if args.hide=="False":
	    # display output
	    cv2.imshow("gender detection", file)

		# press any key to close window           
	    cv2.waitKey()

		# save output
	    cv2.imwrite("gender_detection.jpg", file)

# release resources
cv2.destroyAllWindows()
