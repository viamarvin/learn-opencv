import numpy as np
import cv2

def shapeMatching(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 127, 255, 1)
	cx = 0
	cy = 0

	_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

		if len(approx) == 3:
			shapeName = "Triangle"
			cv2.drawContours(image, [cnt], 0, (0, 255, 0), -1)

			M = cv2.moments(cnt)
			if int(M['m00']) > 0:
				cx = int(M['m10'] / M['m00'])
				cy = int(M['m01'] / M['m00'])
				cv2.putText(image, shapeName, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
		elif len(approx) == 4:
			x, y, w, h = cv2.boundingRect(cnt)
			M = cv2.moments(cnt)
			if int(M['m00']) > 0:
				cx = int(M['m10'] / M['m00'])
				cy = int(M['m01'] / M['m00'])

			if abs(w - h) <= 3:
				shapeName = "Square"

				cv2.drawContours(image, [cnt], 0, (0, 125, 255), -1)
				cv2.putText(image, shapeName, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
			else:
				shapeName = "Rectangle"

				cv2.drawContours(image, [cnt], 0, (0, 0, 255), -1)
				M = cv2.moments(cnt)
				if int(M['m00']) > 0:
					cx = int(M['m10'] / M['m00'])
					cy = int(M['m01'] / M['m00'])
					cv2.putText(image, shapeName, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
		elif len(approx) == 10:
			shapeName = "Star"
			cv2.drawContours(image, [cnt], 0, (255, 255, 0), -1)
			M = cv2.moments(cnt)
			if int(M['m00']) > 0:
				cx = int(M['m10'] / M['m00'])
				cy = int(M['m01'] / M['m00'])
				cv2.putText(image, shapeName, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
		elif len(approx) >= 15:
			shapeName = "Circle"
			cv2.drawContours(image, [cnt], 0, (0, 255, 255), -1)
			M = cv2.moments(cnt)
			if int(M['m00']) > 0:
				cx = int(M['m10'] / M['m00'])
				cy = int(M['m01'] / M['m00'])
				cv2.putText(image, shapeName, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

	return image

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Shape Matching', shapeMatching(frame))

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()