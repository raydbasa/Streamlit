import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO("yolov8m.pt")


def box_to_coord(box):
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    # class_id = result.names[box.cls[0].item()]
    # conf = round(box.conf[0].item(), 2)
    xB = cords[0]
    xA = cords[2]
    yB = cords[1]
    yA = cords[3]
    return xB, xA, yB, yA

def main():
    st.title("YOLO Object Detection with Streamlit")
    st.write("Upload an image and see the detected objects")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # results = model.predict(uploaded_file)
    # result = results[0]

    if uploaded_file is not None:
        # Read the image file
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        # frame = cv2.imdecode(image, 1)  # 1 means load color image
        # results = model.predict(frame)
        # result = results[0]

        # for box in result.boxes:
        #     xB, xA, yB, yA = box_to_coord(box)
        #     cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        st.image(image, channels="BGR")

if __name__ == "__main__":
    main()
