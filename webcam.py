from flask import Flask,render_template,Response
import cv2
from deploy import WebcamDetection

app=Flask(__name__)
# camera=cv2.VideoCapture("https:25.138.100.155:8080/video")


# def generate_frames():
#     while True:
            
#         ## read the camera frame
#         success,frame=camera.read()
#         if not success:
#             break
#         else:
#             ret,buffer=cv2.imencode('.jpg',frame)
#             frame=buffer.tobytes()

#         yield(b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

import torch
import cv2
import time
import time
import numpy as np
capture_index=0    #"https://25.138.100.155:8080/video"
cap=cv2.VideoCapture(capture_index)
model = torch.hub.load('ultralytics/yolov5', 'custom',path="yolo.pt",force_reload=True)
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\n\nDevice Used:",device)
print("-------------------------------------------")
# print(model.eval())
# frame = [frame]
# results = model(frame)
     
# labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

def score_frame(frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        model.to(device)
        frame = [frame]
        results = model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    return classes[int(x)]


def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=cap.read()
        if not success:
            break
        else:
            results = score_frame(frame)
            frame = plot_boxes(results, frame)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('ui.html')

@app.route('/video')
def video():
    # detection = WebcamDetection(capture_index="https:25.138.100.155:8080/video",model_name="best.pt")
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
