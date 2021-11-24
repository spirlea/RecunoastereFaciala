import os
import numpy as np
from PIL import Image
import cv2
import pickle
BASE_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_dir, "imagini")
face_cascade = cv2.CascadeClassifier('Clasificatori/data/haarcascade_frontalface_alt2.xml')
recognition = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
label_ids = {}
x_train = []
y_labels = []
for root,dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            print(label_ids)
            pil_image = Image.open(path).convert('L')
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            fete = face_cascade.detectMultiScale(image_array, 1.3,5)
            for (x, y, w, h) in fete:
                print(x, y, w, h)
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
recognition.train(x_train, np.array(y_labels))
recognition.save("antrenor.yml")
