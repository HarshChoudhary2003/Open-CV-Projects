import cv2
import threading
import yaml
import time
from collections import defaultdict
from detection.yolo import PersonDetector
from tracking.deepsort import Tracker
from reid.reid_model import ReIDModel
from database.db import db

class VideoProcessor:
    def __init__(self, config_path="configs/cameras.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.detector = PersonDetector(model_version="yolov8n.pt")
        self.reid = ReIDModel(threshold=0.85)
        self.running = False
        
        # Track historical centers for drawing paths
        self.track_history = defaultdict(lambda: [])

    def process_camera(self, camera_info):
        cam_id = camera_info["id"]
        cam_url = camera_info["url"]
        tracker = Tracker()
        
        if str(cam_url).isdigit():
            cam_url = int(cam_url)
            
        cap = cv2.VideoCapture(cam_url)
        
        frames_processed = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            frames_processed += 1

            # Step 1: Detect
            detections = self.detector.detect(frame)
            bbs = [(det[0], det[1], det[2]) for det in detections]

            # Step 2: Track
            tracks = tracker.update(bbs, frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                    
                track_id = track.track_id
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Bounding logic
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Step 3: Extract & ReID
                person_crop = frame[y1:y2, x1:x2]
                global_id = f"Ukw_{track_id}"
                
                if person_crop.size > 0:
                    # Every 5 frames update the global embedding to save compute
                    if frames_processed % 5 == 0:
                        embedding = self.reid.extract_features(person_crop)
                        global_id = self.reid.match_identity(embedding, track_id, cam_id)
                        db.add_tracking_event(global_id, cam_id, camera_info["location"])
                        # Cache the global ID in the tracker object for drawing
                        track.global_id = global_id
                    else:
                        global_id = getattr(track, 'global_id', f"Track_{track_id}")

                # Step 4: Draw Advanced Graphics
                # Draw box
                color = (0, 255, 0) if "Person" in global_id else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{global_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Center point for path tracking
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                self.track_history[global_id].append((center_x, center_y))
                
                # Keep only last 30 positions
                if len(self.track_history[global_id]) > 30:
                    self.track_history[global_id].pop(0)
                    
                # Draw trail
                points = self.track_history[global_id]
                for i in range(1, len(points)):
                    thickness = int(np.sqrt(64 / float(len(points) - i + 1)) * 2)
                    cv2.line(frame, points[i - 1], points[i], color, thickness)

            # Save the annotated frame to disk for the dashboard to stream
            try:
                cv2.imwrite("assets/latest_frame.jpg", frame)
            except:
                pass

        cap.release()

    def start(self):
        self.running = True
        self.threads = []
        for cam in self.config["cameras"]:
            t = threading.Thread(target=self.process_camera, args=(cam,))
            t.start()
            self.threads.append(t)

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join()

if __name__ == "__main__":
    import numpy as np
    processor = VideoProcessor()
    processor.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        processor.stop()
