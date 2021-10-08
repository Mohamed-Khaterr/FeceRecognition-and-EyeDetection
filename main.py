import cv2
import face_recognition
import numpy as np
# os to find folder and find images in the folder
import os
from Classes.ImageEncoding import ImageEncoding
from Classes.ReadFrame import ReadFrame

face_cascade = cv2.CascadeClassifier('.xml\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('.xml\haarcascade_eye.xml')

# capture To open Camera in Video
capture = cv2.VideoCapture(0)
# for window width
capture.set(3,1280)
# for window hight
capture.set(4,720)

path = 'Images'
images = []
imageName = []
imageList = os.listdir(path)

for image in imageList:
    currentImage = cv2.imread(f'{path}/{image}')
    images.append(currentImage)
    # to spreate image name from extention
    imageName.append(os.path.splitext(image)[0])
print('Import Images Successfully')

def encodingImage(images=[]):
    encodeList = []
    try:
        for img in images:
            # Convert the image to RGB
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodedImage = face_recognition.face_encodings(image)[0]
            encodeList.append(encodedImage)
        print('Encoding Complete Successfully')
        return encodeList
    except:
        print("Encoding inComplete Error!!: Check if there is Image that hasn't ***Face*** in it")
        pass


encodeListForFaces = encodingImage(images=images)


# Get each frame one by one
while True:
    success, img = capture.read()
    # Resize the image (make image smaller) to decrease Process Time
    imageSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    # Convert the Small image to RGB
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    imageSmall = cv2.cvtColor(imageSmall,cv2.COLOR_BGR2RGB)


    # readObj.calculate(img,encodeListForFaces,imageName)
    # matches, matchIndex = readObj.getMatchAndMatchIndex()



    locationCurrentFrame = face_recognition.face_locations(imageSmall)
    encodeCurrentFrame = face_recognition.face_encodings(imageSmall,locationCurrentFrame)

    for encodeFace, faceLocation in zip(encodeCurrentFrame, locationCurrentFrame):
        # Compare faces
        matches = face_recognition.compare_faces(encodeListForFaces, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListForFaces, encodeFace)

        matchIndex = np.argmin(faceDistance)
        
        # if Face Matches with images in Images Folder
        if matches[matchIndex]:
            name = imageName[matchIndex].upper()
            # To make Rectangle around the Face
            top, right, bottom, left = faceLocation
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
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
    faceImage = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(faceImage, 1.3, 5)
    for x, y, w, z in faces:
        roi_faceImage = faceImage[y:y + z, x:x + w]
        roi_color = img[y:y + z, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_faceImage, 1.3, 5)
        for ex, ey, ew, ez in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + ez), (0, 0, 255), 2)

    cv2.imshow('WebCam ',img)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
capture.release()
cv2.destroyAllWindows()
