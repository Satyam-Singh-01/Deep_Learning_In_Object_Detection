import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_yolo():
    yolo = cv2.dnn.readNet("./yolov3.weights", "./yolov3 (1).cfg")
    classes = []
    with open("./coco.names", 'r') as f:
        classes = f.read().splitlines()
    return yolo, classes

def detect_objects(image, yolo, classes):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)
    output_layer_names = yolo.getUnconnectedOutLayersNames()
    layer_output = yolo.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i], 2))
        color = np.random.uniform(0, 255, size=(3,))
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label + " " + confi, (x, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    return image

def main():
    st.title("Object Detection with YOLO")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = cv2.imread(uploaded_file.name)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        yolo, classes = load_yolo()
        with st.spinner("Detecting objects..."):
            result_image = detect_objects(image, yolo, classes)

        st.subheader("Detection Result:")
        st.image(result_image, caption="Result", use_column_width=True)

if __name__ == "__main__":
    main()