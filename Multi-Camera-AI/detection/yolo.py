from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_version="yolov8n.pt"):
        self.model = YOLO(model_version)

    def detect(self, frame):
        # Run inference
        results = self.model(frame, classes=[0], verbose=False) # class 0 is person
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # Append in format [ [x1, y1, width, height], confidence, class ] for DeepSORT
                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, 0])
                
        return detections
