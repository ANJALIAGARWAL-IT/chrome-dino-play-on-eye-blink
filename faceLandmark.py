import cv2
import dlib

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("C:/Users/hp/Desktop/face_landmarks/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 67):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            #1 for radius of circle
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)#thickness


    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    #Esc key
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()