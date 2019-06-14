import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
	for (x, y, w, h) in faces:
		cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = frame[y:y + h, x:x + w]

		eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
		for (ex, ey, ew, eh) in eyes:
			cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

		smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
		for (sx, sy, sw, sh) in smile:
			cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

	return frame


video_capture = cv.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Face detect', detect(gray, frame))

    if cv.waitKey(1)  & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()