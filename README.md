# 🎥 Open-CV-Projects

<div align="center">

![OpenCV](https://img.shields.io/badge/OpenCV-5C3EEE?style=for-the-badge&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-00A67E?style=for-the-badge)

A comprehensive collection of **computer vision projects** built with OpenCV, demonstrating real-world applications of image processing, object detection, face recognition, gesture recognition, AR filters, and more.

[View Projects](#-projects) • [Installation](#-installation) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Projects](#-projects)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About

This repository contains a curated collection of **21+ computer vision projects** designed to demonstrate practical applications of OpenCV and deep learning. Projects range from basic image processing to advanced AR filters and AI-based systems.

Each project includes:

- ✅ Complete source code
- ✅ Detailed documentation
- ✅ Usage examples
- ✅ Step-by-step guides
- ✅ Real-world applications

---

## 📚 Projects

### **1. Face Detection + Real-time Recognition**
Advanced face detection and recognition system using deep learning models with real-time processing capabilities.

**Features:**
- Real-time face detection in video streams
- Face alignment and normalization
- Facial recognition with high accuracy
- Multi-face tracking
- Expression and emotion detection

### **2. Face Mask Detection**
AI-powered system to detect whether people are wearing masks in images and videos.

**Features:**
- Mask/No-Mask classification
- Real-time video processing
- Confidence score visualization
- Crowd analysis

### **3. AI Virtual Mouse**
Control your cursor using hand gestures and facial movements.

**Features:**
- Hand gesture recognition
- Pointer control via hand movement
- Click/Drag operations using fingers
- Real-time hand tracking

### **4. Face Recognition Attendance System**
Automatic attendance marking system using facial recognition.

**Features:**
- Student/Employee database
- Real-time attendance marking
- Report generation
- Timestamp logging

### **5. Invisible UI Controller**
Control UI elements without visible controllers using hand gestures.

**Features:**
- Gesture-based UI control
- Virtual button interactions
- Real-time hand detection
- Customizable gestures

### **6. Driver Drowsiness & Distraction Detection**
Safety system to detect driver fatigue and distraction in real-time.

**Features:**
- Eye closure detection
- Yawn detection
- Head pose estimation
- Alert system for drowsiness
- Phone usage detection

### **7. YOLO Object Detection**
Real-time object detection using YOLO (You Only Look Once) architecture.

**Features:**
- Multi-object detection
- Real-time processing (30+ FPS)
- Bounding box visualization
- Confidence filtering
- Support for 80+ object classes

### **8. Air Drawing App**
Create drawings in the air using hand movements detected by your webcam.

**Features:**
- Gesture-based drawing
- Color selection
- Brush size adjustment
- Real-time canvas
- Save drawings as images

### **9. Motion-Based Light Control**
Control lights and devices based on motion detection.

**Features:**
- Motion detection
- Light intensity estimation
- Gesture-based control
- IoT integration ready

### **10. Gesture-Based Manual**
Use hand gestures to control presentations and applications.

**Features:**
- Slide navigation gestures
- Volume control gestures
- Pause/Play control
- Custom gesture mapping

### **11. Handwritten Digit Recognition**
Recognize handwritten digits using neural networks trained on MNIST dataset.

**Features:**
- Real-time digit recognition
- Handwriting canvas
- Accuracy metrics
- Model training code included

### **12. AR Face Filters**
Apply Snapchat-like AR filters to faces in real-time.

**Features:**
- Multiple filter effects (sunglasses, hats, masks)
- Real-time face detection
- Smooth filter application
- Customizable filters

### **13. Emotion-Based Color Detection**
Detect emotions from faces and generate corresponding colors.

**Features:**
- 7-emotion classification
- Color mapping for emotions
- Real-time emotion display
- Emotion statistics

### **14. Creative Computer Vision**
Artistic image transformations and creative effects.

**Features:**
- Cartoon effect
- Sketch conversion
- Color manipulation
- Artistic filters

### **15. Gesture-Controlled Presentation**
Control presentations using hand gestures without keyboards.

**Features:**
- Next/Previous slide gestures
- Pointer control
- Zoom functionality
- Real-time gesture feedback

### **16. AR Try-On System**
Virtual try-on system for glasses, hats, and accessories.

**Features:**
- Real-time object placement
- Accurate face alignment
- Multiple product catalog
- Rotation and scaling

### **17. AI Sign Language Translator**
Translate sign language to text and speech in real-time.

**Features:**
- Hand pose estimation
- Gesture recognition
- Real-time translation
- Text-to-speech output

### **18. Real-Time Background Removal**
Remove or replace backgrounds in real-time video streams.

**Features:**
- Semantic segmentation
- Custom background replacement
- Virtual backgrounds
- Smooth edge blending

### **19. Motion-Driven Generation**
Generate images or effects based on motion patterns.

**Features:**
- Motion tracking
- Dynamic effect generation
- Frame interpolation
- Creative visualizations

### **20. AI Attention & Focus Monitor**
Monitor attention level in meetings and classes.

**Features:**
- Gaze tracking
- Head position monitoring
- Distraction detection
- Attention reports

### **21. Hand-Controlled 3D Navigation**
Navigate 3D environments using hand gestures.

**Features:**
- Hand gesture recognition
- 3D object rotation
- Zoom and pan control
- Real-time tracking

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Virtual environment (recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/HarshChoudhary2003/Open-CV-Projects.git
cd Open-CV-Projects
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Or using conda
conda create -n cv-projects python=3.9
conda activate cv-projects
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
```
opencv-python==4.8.0
opencv-contrib-python==4.8.0
numpy==1.24.3
scipy==1.11.1
scikit-image==0.21.0
Pillow==10.0.0
matplotlib==3.7.2
tensorflow==2.13.0
torch==2.0.0
torchvision==0.15.0
pandas==2.0.3
medapipe==0.10.0
```

---

## 💻 Usage

### Running Individual Projects

Each project can be run independently:

```bash
# Example: Running Face Detection
cd "Project 1-- Face Detection + Rea"
python face_detection.py

# Example: Running YOLO Object Detection
cd "Project 7--Object_Detection_YO"
python object_detection.py
```

### Example Code

```python
import cv2

# Load video
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process frame (add your project logic here)
    processed = frame  # Your processing
    
    cv2.imshow('Output', processed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 Technology Stack

- **Python 3.9+** - Core programming language
- **OpenCV 4.8** - Computer vision library
- **TensorFlow/Keras** - Deep learning framework
- **PyTorch** - ML framework for some projects
- **MediaPipe** - Hand and pose detection
- **NumPy** - Numerical computing
- **Scikit-image** - Image processing
- **Matplotlib** - Visualization

---

## 📁 Project Structure

```
Open-CV-Projects/
├── Project 1-- Face Detection + Rea.../
├── Project 2-- Face Mask Detection/
├── Project 3-- AI Virtual Mouse/
├── Project 4-- Face Recognition Att.../
├── Project 5-- Invisible UI Controller/
├── Project 6-- Driver Drowsiness & .../
├── Project 7--Object_Detection_YO.../
├── Project 8-- Air Drawing App/
├── Project 9-- Motion-Based Light .../
├── Project 10-- Gesture-Based Man.../
├── Project 11-- Handwritten Digit.../
├── Project 12--AR Face Filters/
├── Project 13-- Emotion-Based Col.../
├── Project 14-- Creative Computer V.../
├── Project 15-- Gesture-Controlled.../
├── Project 16-- AR Try-On System/
├── Project 17-- AI Sign Language Tr.../
├── Project 18-- Real-Time Backgrou.../
├── Project 19-- Motion-Driven Gen.../
├── Project 20-- AI Attention & Focu.../
├── Project 21-- Hand-Controlled 3.../
├── README.md
└── LICENSE
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Harsh Choudhary**
- 🔗 GitHub: [@HarshChoudhary2003](https://github.com/HarshChoudhary2003)
- 💼 LinkedIn: [Harsh Choudhary](https://linkedin.com/in/harshchoudhary2003)
- 📧 Email: hc504360@gmail.com

---

## 🌟 Show Your Support

If this repository helped you, please give it a ⭐ and share with others!

---

<div align="center">

**Made with ❤️ by Harsh Choudhary**

⬆️ [Back to Top](#-open-cv-projects)

</div>
