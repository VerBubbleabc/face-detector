import numpy as np
import cv2
import matplotlib.pyplot as plt

def convertToRGB(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_img(img, save=False):
	plt.imshow(convertToRGB(img))
	if save: plt.savefig('test.jpg')
	plt.show()

def detect_faces(f_cascade, colored_img, scaleFactor=1.2):
	img_copy = np.copy(colored_img)
	gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
	faces  =f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
	for (x, y, w, h) in faces:
		cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
	return img_copy, faces

def main():
	haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
	test = cv2.imread('data/test1.jpg')
	faces_detected_img, faces = detect_faces(haar_face_cascade, test)
	show_img(faces_detected_img, True)

if __name__ == '__main__':
	main()