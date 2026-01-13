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

This repository contains a curated collection of **32+ computer vision projects** designed to demonstrate practical applications of OpenCV and deep learning. Projects range from basic image processing to advanced AR filters and AI-based systems.

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

### **2. Face Mask Detection**
AI-powered system to detect whether people are wearing masks in images and videos.

### **3. AI Virtual Mouse**
Control your cursor using hand gestures and facial movements.

### **4. Face Recognition Attendance System**
Automatic attendance marking system using facial recognition.

### **5. Invisible UI Controller**
Control UI elements without visible controllers using hand gestures.

### **6. Driver Drowsiness & Distraction Detection**
Safety system to detect driver fatigue and distraction in real-time.

### **7. YOLO Object Detection**
Real-time object detection using YOLO (You Only Look Once) architecture.

### **8. Air Drawing App**
Create drawings in the air using hand movements detected by your webcam.

### **9. Motion-Based Light Control**
Control lights and devices based on motion detection.

### **10. Gesture-Based Manual**
Use hand gestures to control presentations and applications.

### **11. Handwritten Digit Recognition**
Recognize handwritten digits using neural networks trained on MNIST dataset.

### **12. AR Face Filters**
Apply Snapchat-like AR filters to faces in real-time.

### **13. Emotion-Based Color Detection**
Detect emotions from faces and generate corresponding colors.

### **14. Creative Computer Vision**
Artistic image transformations and creative effects.

### **15. Gesture-Controlled Presentation**
Control presentations using hand gestures without keyboards.

### **16. AR Try-On System**
Virtual try-on system for glasses, hats, and accessories.

### **17. AI Sign Language Translator**
Translate sign language to text and speech in real-time.

### **18. Real-Time Background Removal**
Remove or replace backgrounds in real-time video streams.

### **19. Motion-Driven Generation**
Generate images or effects based on motion patterns.

### **20. AI Attention & Focus Monitor**
Monitor attention level in meetings and classes.

### **21. Hand-Controlled 3D Navigation**
Navigate 3D environments using hand gestures.

### **22. AI Emotion Mirror**
Interactive mirror that reflects and enhances your emotions using AI.

### **23. Time-Echo Camera (Motion Ghosts)**
Visualizes motion history using long-exposure and frame interpolation effects.

### **24. AI Shadow Art Generator**
Converts real-time shadows into artistic representations or interactive art.

### **25. Motion-to-Music Painter**
Generates musical notes and visual art based on body movements.

### **26. Reality Distortion Field**
Applies mind-bending mathematical distortions to real-time video feeds.

### **27. AI Aura Visualizer**
Visualizes "energy fields" around humans using pose estimation and color mapping.

### **28. Living Portrait Generator**
Animates static portraits with realistic facial movements and expressions.

### **29. Interactive Kaleidoscope World**
Transform webcam feeds into dynamic, interactive kaleidoscopic patterns.

### **30. AI Dream Camera**
Applies neural style transfer and deep dream effects to real-time video.

### **31. Advanced 3D Air Drawing System**
A sophisticated system for 3D sketching and modeling in thin air.

### **32. Jarvis-Style 3D Holographic HUD**
A high-tech heads-up display inspired by Iron Man's Jarvis interface.

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
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cv-projects python=3.9
conda activate cv-projects
```

### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install opencv-python opencv-contrib-python numpy mediapipe tensorflow torch
```

---

## 💻 Usage

### Running Individual Projects

Each project folder contains its own execution script. For example:

```bash
# Running Project 32
cd "Project 32-- JARVIS-STYLE 3D HOLOGRAPHIC HUD"
python jarvis_hud.py
```

---

## 📁 Project Structure

```
Open-CV-Projects/
├── Project 1 --Face Detection + Real-Time Camera/
├── Project 2-- Face Mask Detection/
├── Project 3-- AI Virtual Mouse/
├── Project 4-- Face Recognition Attendance System/
├── Project 5-- Invisible UI Controller/
├── Project 6-- Driver Drowsiness & Distraction Detection System/
├── Project 7--Object_Detection_YOLO/
├── Project 8-- Air Drawing App/
├── Project 9-- Motion-Based Light Painting/
├── Project 10-- Gesture-Based Mandala/
├── Project 11-- Handwritten Digit Drawing/
├── Project 12--AR Face Filters/
├── Project 13-- Emotion-Based Color Painting/
├── Project 14--Creative Computer Vision Studio/
├── Project 15-- Gesture-Controlled Music Visualizer/
├── Project 16-- AR Try-On System/
├── Project 17-- AI Sign Language Translator/
├── Project 18-- Real-Time Background Replacement/
├── Project 19-- Motion-Driven Generative Art Engine/
├── Project 20-- AI Attention & Focus Detector/
├── Project 21-- Hand-Controlled 3D Object Manipulation/
├── Project 22--AI Emotion Mirror/
├── Project 23-- Time-Echo Camera (Motion Ghosts)/
├── Project 24-- AI Shadow Art Generator/
├── Project 25-- Motion-to-Music Painter/
├── Project 26-- Reality Distortion Field/
├── Project 27-- AI Aura Visualizer/
├── Project 28-- Living Portrait Generator/
├── Project 29-- Interactive Kaleidoscope World/
├── Project 30-- AI Dream Camera/
├── Project 31--ADVANCED 3D AIR DRAWING SYSTEM/
└── Project 32-- JARVIS-STYLE 3D HOLOGRAPHIC HUD/
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

This project is licensed under the **MIT License**.

---

## 👤 Author

**Harsh Choudhary**
- 🔗 GitHub: [@HarshChoudhary2003](https://github.com/HarshChoudhary2003)
- 💼 LinkedIn: [Harsh Choudhary](https://linkedin.com/in/harshchoudhary2003)
- 📧 Email: hc504360@gmail.com

---

<div align="center">

**Made with ❤️ by Harsh Choudhary**

⬆️ [Back to Top](#-open-cv-projects)

</div>
