import cv2
import face_recognition
import numpy as np


class ReadFrame:
	face_cascade = cv2.CascadeClassifier('.xml\haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('.xml\haarcascade_eye.xml')

	match = False
	matchIndex = []

	# def __init__(self,frame,encode):
	#     self.img = frame
	#     self.encodeListForFaces = encode

	def calculate(self, img, encodeListForFaces,imageName):
		# Resize the image (make image smaller) to decrease Process Time
		imageSmall = cv2.resize(img,(0, 0),None,0.25,0.25)
		# Convert the Small image to RGB
		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		imageSmall = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2RGB)

		locationCurrentFrame = face_recognition.face_locations(imageSmall)
		encodeCurrentFrame = face_recognition.face_encodings(imageSmall, locationCurrentFrame)

		for encodeFace, faceLocation in zip(encodeCurrentFrame, locationCurrentFrame):
			# Compare faces
			matches = face_recognition.compare_faces(encodeListForFaces, encodeFace)
			#faceDistance = face_recognition.face_distance(encodeListForFaces, encodeFace)

			#matchIndex = np.argmin(faceDistance)

			#self.setMatchAndMatchIndex(m=matches,mi=matchIndex)


			# if Face Matches with images in Images Folder
			# if matches[matchIndex]:
			# 	name = imageName[matchIndex].upper()
			# 	# To make Rectangle around the Face
			# 	top, right, bottom, left = faceLocation
			# 	top *= 4
			# 	right *= 4
			# 	bottom *= 4
			# 	left *= 4
			# 	cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
			# 	# Draw a label with a name below the face
			# 	cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
			# 	font = cv2.FONT_HERSHEY_DUPLEX
			# 	cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
			# else:
			top, right, bottom, left = faceLocation
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4
			cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
			# Draw a label with a name below the face
			cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(img, 'Unkown', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


		# For Eye Detection
		# Can't resize image
		faceImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		faces = self.face_cascade.detectMultiScale(faceImage, 1.3, 5)
		for x, y, w, z in faces:
			roi_faceImage = faceImage[y:y + z, x:x + w]
			roi_color = img[y:y + z, x:x + w]
			eyes = self.eye_cascade.detectMultiScale(roi_faceImage, 1.3, 5)
			for ex, ey, ew, ez in eyes:
				cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + ez), (0, 0, 255), 2)


	def setMatchAndMatchIndex(self,m,mi):
		self.match = m
		self.matchIndex = mi

	def getMatchAndMatchIndex(self):
		return self.match, self.matchIndex