import cv2
import numpy as np
import face_recognition
import os
import tablib
from datetime import datetime
from flask import Flask, render_template, Response, request
import csv
import pandas as pd

app = Flask(__name__)

path = 'ImagesAttendance'
images = []
classNames = []
nameList = []
myList = os.listdir(path)
# print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAbsent(name1):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        status = "Absent"
        now = datetime.now()
        dtString = now.strftime("%d-%m-%Y %H:%M:%S")
        f.writelines(f'\n{name1}, {"00-00-0000 00:00:00"}, {status}')


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList1 = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList1:
            now = datetime.now()
            dtString = now.strftime("%d-%m-%Y %H:%M:%S")
            status = "Present"
            f.writelines(f'\n{name}, {dtString}, {status}')
            print(name + "   -\t" + 'Attendance Marked')


encodeListKnown = findEncodings(images)
# print('Encoding Complete')


def gen_frames():
    cap = cv2.VideoCapture(0)

    for i in range(len(classNames)):
        abs_name = classNames[i].upper()
        print(abs_name)
        if abs_name not in nameList:
            markAbsent(abs_name)

    while True:
        success, img = cap.read()
        # img = captureScreen()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] < 0.5:
                name = classNames[matchIndex].upper()
                markAttendance(name)
            else:
                name = 'Unknown'
            # print(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Detection', img)
        cv2.waitKey(1)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        results = []

        with open('Attendance.csv') as csvfile:
            csvReader = csv.DictReader(csvfile)
            for row in csvReader:
                results.append(dict(row))
        # print(results)

        fieldnames = [key for key in results[0].keys()]
        return render_template('index.html', results=results, fieldnames=fieldnames, len=len)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
