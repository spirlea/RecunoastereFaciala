import cv2
import numpy as np
import pickle
clasificator = cv2.CascadeClassifier('Clasificatori/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("antrenor.yml")
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}
cap = cv2.VideoCapture(0)
print(cv2.__file__)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fete = clasificator.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in fete:
        print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]
        img_item = "imaginea_mea.png"
        cv2.imwrite(img_item, roi_gray)
        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        color = (255, 0, 0)
        grosime = 2
        coord_x = x + w
        coord_y = y + h
        cv2.rectangle(frame, (x, y), (coord_x, coord_y), color, grosime)
    cv2.imshow("Camera", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
