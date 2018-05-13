from statistics import mode

import cv2
from keras.models import load_model
import numpy as np


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)



detection_model_path = './models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './models/emotion_models/model_resnet.102-0.66.hdf5'				## change according to the model file name
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
frame_window = 10
emotion_offsets = (20, 40)

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

emotion_target_size = emotion_classifier.input_shape[1:3]

emotion_window = []

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
  feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
	bgr_image = video_capture.read()[1]
	gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
	rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
	faces = face_detection.detectMultiScale(gray_image, 1.3, 5)

	for face_coordinates in faces:
		
		x, y, width, height = face_coordinates
		x_off, y_off = emotion_offsets
		x1, x2, y1, y2 = x - x_off, x + width + x_off, y - y_off, y + height + y_off

		gray_face = gray_image[y1:y2, x1:x2]
		try:
		    gray_face = cv2.resize(gray_face, (emotion_target_size))
		except:
		    continue

		gray_face = gray_face.astype('float32')
		gray_face = gray_face / 255.0
		gray_face = gray_face - 0.5
		gray_face = gray_face * 2.0
		gray_face = np.expand_dims(gray_face, 0)
		gray_face = np.expand_dims(gray_face, -1)

		emotion_prediction = emotion_classifier.predict(gray_face)
		emotion_probability = np.max(emotion_prediction)
		emotion_label_arg = np.argmax(emotion_prediction)
		emotion_text = emotion_labels[emotion_label_arg]
		emotion_window.append(emotion_text)

		if len(emotion_window) > frame_window:
		    emotion_window.pop(0)
		try:
		    emotion_mode = mode(emotion_window)
		except:
		    continue

		if emotion_text == 'angry':
		    color = emotion_probability * np.asarray((255, 0, 0))
		elif emotion_text == 'sad':
		    color = emotion_probability * np.asarray((0, 0, 255))
		elif emotion_text == 'happy':
		    color = emotion_probability * np.asarray((255, 255, 0))
		elif emotion_text == 'surprise':
		    color = emotion_probability * np.asarray((0, 255, 255))
		else:
		    color = emotion_probability * np.asarray((0, 255, 0))

		color = color.astype(int)
		color = color.tolist()

		x, y, w, h = face_coordinates
		cv2.rectangle(rgb_image, (x, y), (x + w, y + h), color, 2)
		draw_text(face_coordinates, rgb_image, emotion_mode,
		          color, 0, -45, 1, 1)

	
		face_image = feelings_faces[emotion_label_arg]
		for c in range(0, 3):
			rgb_image[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  rgb_image[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
        
	bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
	cv2.imshow('window_frame', bgr_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break


