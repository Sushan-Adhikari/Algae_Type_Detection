from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from typing import List

# Initialize the FastAPI app
app = FastAPI()

# Load YOLOv8 model
model = YOLO("best_yolov8m_model.pt")

# Class names for algae
class_names = ["blue-green-algae", "brown-algae", "red-algae"]

def predict(image: Image.Image):
    # Convert image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Perform inference
    results = model(image_cv)

    # Process the results
    detected_objects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            conf = box.conf
            bbox = box.xyxy
            detected_objects.append({
                "class": class_names[class_id],
                "confidence": float(conf),
                "bbox": bbox.tolist()
            })
    
    return detected_objects, results

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    # Read image from the uploaded file
    image = Image.open(io.BytesIO(await file.read()))

    # Run inference
    detections, results = predict(image)

    # Annotate the image with bounding boxes
    result_image = results[0].plot()  # Plot the bounding boxes

    # Convert to bytes to return as response
    _, encoded_image = cv2.imencode('.jpg', result_image)
    result_image_bytes = encoded_image.tobytes()

    return {
        "image": result_image_bytes,  # Encoded result image with bounding boxes
        "detections": detections  # List of detected algae types and bounding boxes
    }
