import os
import cv2
import face_recognition


class ImageEncoding:

	# path to Images folder
	path = ''
	images = []
	imageName = []
	imageList = []

	def __init__(self,path):
		self.path = path
		# to store every image in Image folder in list called (imageList)
		self.imageList = os.listdir(path)

	def readImages(self):
		# Read Image from path Directory
		for image in self.imageList:
			currentImage = cv2.imread(f'{self.path}/{image}')
			self.images.append(currentImage)
			# to spreate image name from extention
			self.imageName.append(os.path.splitext(image)[0])
		print('Import Images Successfully')

	# function To Encode all Images
	# All images must contain face or it will be error (index out of range)
	# def encodingImage(images):
	def encodingImage(self):
		encodeList = []
		try:
			for img in self.images:
				# Convert the image to RGB
				image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				encodedImage = face_recognition.face_encodings(image)[0]
				encodeList.append(encodedImage)
			print('Encoding Complete Successfully')
			return encodeList
		except:
			print("Encoding inComplete Error!!: Check if there is Image that hasn't ***Face*** in it")
			pass

	def getImageName(self):
		return self.imageName