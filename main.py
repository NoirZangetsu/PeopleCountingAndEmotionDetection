import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import time
import requests
import json
from deepface import DeepFace

cred = credentials.Certificate('amal-288d9-firebase-adminsdk-9dbiw-1a8900cf76.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# YOLO modelini yükleme
modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hava kalitesi verilerini çeken fonksiyon
def get_ozone_level(country_code):
    url = f"https://api.openaq.org/v2/locations?country={country_code}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    air_quality_data = json.loads(response.content)
    ozone_data = next((item for item in air_quality_data['results'][0]['parameters'] if item["parameter"] == "o3"), None)
    return ozone_data['lastValue'] if ozone_data else None

def update_firestore(people_count, emotions, ages, ozone_level):
    doc_ref = db.collection('people_count').document('current_count')
    doc_ref.set({
        'people_count': people_count,
        'emotions': emotions,
        'ages': ages,
        'ozone_level': ozone_level,
        'timestamp': datetime.datetime.now().isoformat()
    })

def get_human_boxes(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5 and classID == 0:  # classID 0, COCO datasette 'person' sınıfına karşılık gelir
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    if isinstance(idxs, tuple):
        idxs = idxs[0] if len(idxs) > 0 else []
    return idxs, boxes

cap = cv2.VideoCapture(0)
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in (unconnected_layers.flatten())]
last_saved_time = time.time()
last_people_count = -1
last_emotions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti için Haar Cascade kullanma
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # YOLO ile insan tespiti
    idxs, boxes = get_human_boxes(frame, net, output_layers)
    people_count = len(idxs)  # YOLO ile tespit edilen insan sayısı

    emotions = []
    ages = []

    # Yüzler için duygu ve yaş analizi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_frame = frame[y:y + h, x:x + w]

        try:
            results = DeepFace.analyze(face_frame, actions=['emotion', 'age'], enforce_detection=False)
            if isinstance(results, list):
                for result in results:
                    dominant_emotion = result.get('dominant_emotion')
                    age = result.get('age')
                    if dominant_emotion and age:
                        emotions.append(dominant_emotion)
                        ages.append(age)
            else:
                emotions.append(results['dominant_emotion'])
                ages.append(results['age'])
        except Exception as e:
            print(f"Analiz hatası: {e}")

    # YOLO ile tespit edilen insanları çerçeveleme
    if len(idxs) > 0:
        for i in idxs:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    ozone_level = get_ozone_level("ME")  # Örnek olarak "ME" (Karadağ) kullanıldı

    if (time.time() - last_saved_time > 30) or (people_count != last_people_count) or (emotions != last_emotions):
        last_saved_time = time.time()
        last_people_count = people_count
        last_emotions = emotions.copy()  # Son duyguları kaydet
        update_firestore(people_count, emotions, ages, ozone_level)
        print(f"Updated people count: {people_count}, Emotions: {emotions}, Ages: {ages}")

    cv2.imshow('Kamera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
