<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,50:00D4FF,100:00FF88&height=220&section=header&text=Open%20CV%20Projects&fontSize=55&fontColor=ffffff&fontAlignY=38&desc=32%2B%20Computer%20Vision%20Projects%20%7C%20OpenCV%20%7C%20Deep%20Learning%20%7C%20AI&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<!-- Animated Typing SVG -->
<a href="https://github.com/HarshChoudhary2003/Open-CV-Projects">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=00D4FF&center=true&vCenter=true&multiline=true&width=700&height=60&lines=🎥+Computer+Vision+%7C+AI+%7C+Deep+Learning;🤖+32%2B+Projects+from+Beginner+to+Advanced;🚀+OpenCV+%7C+MediaPipe+%7C+YOLO+%7C+TensorFlow" alt="Typing SVG" />
</a>

<br/>
<br/>

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EEE?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Solutions-00897B?style=for-the-badge&logo=google&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v8-FF004F?style=for-the-badge&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Projects](https://img.shields.io/badge/Projects-32%2B-blueviolet?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/HarshChoudhary2003/Open-CV-Projects?style=for-the-badge&color=gold)

<br/>

<!-- Navigation -->
[🎯 About](#-about) &nbsp;•&nbsp;
[📚 Projects](#-all-32-projects) &nbsp;•&nbsp;
[🛠 Tech Stack](#-technology-stack) &nbsp;•&nbsp;
[⚙️ Installation](#-installation) &nbsp;•&nbsp;
[📁 Structure](#-project-structure) &nbsp;•&nbsp;
[🤝 Contributing](#-contributing) &nbsp;•&nbsp;
[👤 Author](#-author)

</div>

---

## 🎯 About

<img align="right" src="https://raw.githubusercontent.com/devSouvik/devSouvik/master/gif3.gif" width="280"/>

**Open-CV-Projects** is a flagship collection of **32+ production-ready computer vision applications** built with OpenCV, MediaPipe, TensorFlow, YOLO, and modern deep learning frameworks. This repository takes you from foundational real-time detection all the way to cutting-edge holographic HUDs, generative AI art engines, and immersive AR experiences.

### ✨ What Makes This Special?
- 🔬 **Beginner → Advanced** — progressive difficulty across 32 projects
- 🧠 **AI-Powered** — deep learning models integrated throughout
- 🎨 **Creative Vision** — art generation, aura visualization, dream cameras
- ⚡ **Real-Time** — optimized for live webcam performance
- 🗂️ **Well Documented** — every project is self-contained and explained

<br clear="right"/>

---

## 📊 Stats at a Glance

<div align="center">

| 📦 Total Projects | 🧠 AI/DL Projects | 🎨 Creative Projects | 🤖 AR Projects | ⚡ Real-Time |
|:-:|:-:|:-:|:-:|:-:|
| **32+** | **18** | **10** | **4** | **All** |

</div>

---

## 📚 All 32 Projects

> 🔗 **Click any project name to navigate directly to its folder!**

---

### 🔵 TIER 1 — Foundations & Detection

<details>
<summary><b>📁 Click to expand Tier 1 Projects</b></summary>
<br/>

#### 1. 🎭 [Face Detection + Real-Time Camera](./Face%20Detection%20+%20Real-Time%20Camera/)

> **Difficulty:** 🟢 Beginner &nbsp;|&nbsp; **Tech:** OpenCV, Haar Cascades, DNN &nbsp;|&nbsp; **Use Case:** Security, Surveillance

Real-time face detection using OpenCV's deep neural network (DNN) module and Haar cascade classifiers. Processes live webcam feed at 30+ FPS with bounding box overlays, confidence scores, and multi-face support.

**Key Features:**
- ✅ Multi-face simultaneous detection
- ✅ DNN-based high accuracy model
- ✅ Real-time FPS display
- ✅ Adjustable confidence threshold

**Tech Stack:** `cv2.dnn` · `imutils` · `numpy`

---

#### 2. 😷 [Face Mask Detection](./Face%20Mask%20Detection/)

> **Difficulty:** 🟢 Beginner &nbsp;|&nbsp; **Tech:** TensorFlow/Keras, MobileNetV2 &nbsp;|&nbsp; **Use Case:** Public Safety, COVID-19

AI-powered binary classifier that detects mask/no-mask on human faces in real-time. Built on a fine-tuned MobileNetV2 backbone for lightweight yet accurate inference on live video streams.

**Key Features:**
- ✅ Binary classification (Mask / No Mask)
- ✅ MobileNetV2 transfer learning
- ✅ Color-coded bounding boxes (Green/Red)
- ✅ Multi-person simultaneous detection

**Tech Stack:** `tensorflow` · `keras` · `sklearn` · `opencv`

---

#### 3. 🖱️ [AI Virtual Mouse](./AI%20Virtual%20Mouse/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** MediaPipe Hands, PyAutoGUI &nbsp;|&nbsp; **Use Case:** Accessibility, Touchless Control

Control your entire computer cursor with your index finger through a webcam. Implements gesture recognition for left click, right click, scroll, and drag — completely hands-free.

**Key Features:**
- ✅ Index finger cursor tracking
- ✅ Pinch gesture for click
- ✅ Scroll gesture support
- ✅ Smoothing filters to eliminate jitter

**Tech Stack:** `mediapipe` · `pyautogui` · `opencv` · `numpy`

---

#### 4. 🪪 [Face Recognition Attendance System](./Face%20Recognition%20Attendance%20System/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** face_recognition, dlib, CSV &nbsp;|&nbsp; **Use Case:** Education, Corporate

Automated attendance management system that identifies registered faces in real-time and logs entries with timestamps to a CSV file. Supports adding new students via webcam enrollment.

**Key Features:**
- ✅ Face enrollment and recognition pipeline
- ✅ Automatic CSV attendance logging
- ✅ Timestamp with date/time per entry
- ✅ Unknown face detection fallback

**Tech Stack:** `face_recognition` · `dlib` · `opencv` · `pandas`

---

#### 5. 👻 [Invisible UI Controller](./Invisible%20UI%20Controller/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** MediaPipe, Color Segmentation &nbsp;|&nbsp; **Use Case:** Touchless Interface

Inspired by the famous "invisible cloak" effect — control UI elements using a specific-colored object or bare hand without any physical hardware. The hand becomes the controller.

**Key Features:**
- ✅ Color-masking based object tracking
- ✅ Gesture-to-UI event mapping
- ✅ Region-based interaction zones
- ✅ Works under varied lighting

**Tech Stack:** `opencv` · `mediapipe` · `numpy`

---

#### 6. 🚗 [Driver Drowsiness & Distraction Detection System](./Driver%20Drowsiness%20%26%20Distraction%20Detection%20System/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** dlib, EAR/MAR Algorithm &nbsp;|&nbsp; **Use Case:** Road Safety, Automotive

Real-time driver safety system that monitors Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to detect drowsiness and yawning. Triggers audio alerts when dangerous states are detected.

**Key Features:**
- ✅ Eye Aspect Ratio (EAR) drowsiness detection
- ✅ Mouth Aspect Ratio (MAR) yawning detection
- ✅ Audio alarm trigger system
- ✅ Frame-counter based alerting

**Tech Stack:** `dlib` · `scipy` · `imutils` · `pygame`

---

#### 7. 🔍 [Object Detection — YOLO](./Object_Detection_YOLO/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** YOLOv8, Ultralytics &nbsp;|&nbsp; **Use Case:** Robotics, Surveillance, Industry

Real-time multi-class object detection using the YOLO (You Only Look Once) architecture. Detects 80+ object classes from the COCO dataset with bounding boxes, labels, and confidence scores.

**Key Features:**
- ✅ 80+ class COCO dataset detection
- ✅ Real-time webcam + video file support
- ✅ Confidence threshold filtering
- ✅ NMS post-processing

**Tech Stack:** `ultralytics` · `yolov8` · `opencv` · `torch`

---

#### 8. ✏️ [Air Drawing App](./Air%20Drawing%20App/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** MediaPipe Hands, Canvas overlay &nbsp;|&nbsp; **Use Case:** Digital Art, Education

Draw freely in the air using your index finger as a virtual brush. Features a color palette, adjustable brush sizes, eraser mode, and canvas clear — all controlled through gestures.

**Key Features:**
- ✅ Index finger drawing tracking
- ✅ Multi-color palette selection
- ✅ Brush size control via gestures
- ✅ Eraser and canvas clear mode

**Tech Stack:** `mediapipe` · `opencv` · `numpy`

</details>

---

### 🟡 TIER 2 — Creative & Interactive

<details>
<summary><b>📁 Click to expand Tier 2 Projects</b></summary>
<br/>

#### 9. 💡 [Motion-Based Light Painting](./Motion-Based%20Light%20Painting/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** Optical Flow, Frame Accumulation &nbsp;|&nbsp; **Use Case:** Photography, Art

Mimics long-exposure light painting photography using motion trails from real-time webcam feed. Colored motion traces accumulate over time creating stunning light art effects.

**Key Features:**
- ✅ Optical flow motion tracking
- ✅ Long-exposure frame accumulation
- ✅ Multi-color trail modes
- ✅ Save canvas as image

**Tech Stack:** `opencv` · `numpy`

---

#### 10. 🌸 [Gesture-Based Mandala — Generative Art](./Gesture-Based%20Mandala%20--%20Generative%20Art/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** MediaPipe, Trigonometry &nbsp;|&nbsp; **Use Case:** Generative Art, Creative Tools

Generates symmetrical mandala patterns in real-time by tracking hand gestures. Movement speed, angle, and position map to pattern density, color shifts, and rotation.

**Key Features:**
- ✅ Symmetrical radial mandala generation
- ✅ Gesture-driven color mapping
- ✅ Dynamic speed-to-density mapping
- ✅ Export mandala as PNG

**Tech Stack:** `mediapipe` · `opencv` · `math` · `numpy`

---

#### 11. 🔢 [Handwritten Digit Drawing — AI Prediction](./Handwritten%20Digit%20Drawing%20--AI%20Prediction/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** TensorFlow, MNIST, Canvas &nbsp;|&nbsp; **Use Case:** Education, ML Demo

Draw digits in the air or on-screen, and a trained CNN model predicts the digit in real-time. Built on the MNIST dataset with live confidence bars per digit class.

**Key Features:**
- ✅ Draw digits via finger or mouse
- ✅ CNN model trained on 60K MNIST samples
- ✅ Live confidence bar per class (0–9)
- ✅ Auto-center and normalize input

**Tech Stack:** `tensorflow` · `keras` · `opencv` · `numpy`

---

#### 12. 🎭 [AR Face Filters](./AR%20Face%20Filters/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** dlib, facial landmarks &nbsp;|&nbsp; **Use Case:** Social Media, Entertainment

Apply Snapchat-style AR filters in real-time: glasses, hats, beards, dog ears, and more. Filters are precisely aligned using 68-point facial landmark detection.

**Key Features:**
- ✅ 68-point dlib landmark detection
- ✅ Multiple switchable filter overlays
- ✅ Alpha-blended PNG composition
- ✅ Head-pose aware filter rotation

**Tech Stack:** `dlib` · `opencv` · `numpy` · `PIL`

---

#### 13. 🎨 [Emotion-Based Color Painting](./Emotion-Based%20Color%20Painting/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** DeepFace / FER, Canvas &nbsp;|&nbsp; **Use Case:** Emotion AI, Art Therapy

Detects your facial emotion (happy, sad, angry, surprise, fear, disgust, neutral) and maps it to a corresponding color palette that dynamically paints the canvas.

**Key Features:**
- ✅ 7-class facial emotion recognition
- ✅ Emotion-to-color palette mapping
- ✅ Smooth color interpolation transitions
- ✅ Live emotion confidence display

**Tech Stack:** `deepface` · `opencv` · `numpy`

---

#### 14. 🎬 [Creative Computer Vision Studio (All-in-One App)](./Creative%20Computer%20Vision%20Studio%20%28All-in-One%20App%29/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** OpenCV, Multiple Models &nbsp;|&nbsp; **Use Case:** CV Showcase, Demo

An all-in-one studio combining 10+ computer vision effects: cartoonization, pencil sketch, edge detection, oil painting, color splash, emboss, and more — switchable in real-time.

**Key Features:**
- ✅ 10+ live visual effect modes
- ✅ Keyboard shortcut switching
- ✅ Record output video
- ✅ Side-by-side before/after view

**Tech Stack:** `opencv` · `numpy` · `PIL`

---

#### 15. 🎵 [Gesture-Controlled Music Visualizer](./Gesture-Controlled%20Music%20Visualizer/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** MediaPipe, Pygame, FFT &nbsp;|&nbsp; **Use Case:** Music, Creative

Control music playback and visualizations using hand gestures. Volume, track switching, and visual effects (spectrum, waveform, particles) are fully gesture-controlled.

**Key Features:**
- ✅ FFT-based audio spectrum visualization
- ✅ Hand gesture music control
- ✅ Multiple visualizer modes
- ✅ BPM sync animations

**Tech Stack:** `mediapipe` · `pygame` · `numpy` · `librosa`

---

#### 16. 👗 [AR Try-On System](./AR%20Try-On%20System/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** MediaPipe Face Mesh, Pose &nbsp;|&nbsp; **Use Case:** eCommerce, Fashion

Virtual augmented reality try-on for glasses, hats, necklaces, and upper body clothing. Items warp and resize accurately with head pose and body position in real-time.

**Key Features:**
- ✅ Face mesh + pose landmark alignment
- ✅ Head-pose adaptive rotation/scaling
- ✅ Multi-item wardrobe switching
- ✅ Photorealistic alpha compositing

**Tech Stack:** `mediapipe` · `opencv` · `numpy` · `PIL`

</details>

---

### 🔴 TIER 3 — AI & Advanced Systems

<details>
<summary><b>📁 Click to expand Tier 3 Projects</b></summary>
<br/>

#### 17. 🤟 [AI Sign Language Translator](./AI%20Sign%20Language%20Translator/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** MediaPipe Hands, LSTM, TTS &nbsp;|&nbsp; **Use Case:** Accessibility, Communication

Translates ASL (American Sign Language) hand gestures to text and speech in real-time using an LSTM sequence model trained on landmark trajectories.

**Key Features:**
- ✅ 26 ASL alphabet + word-level signs
- ✅ LSTM temporal sequence recognition
- ✅ Text-to-speech output (pyttsx3)
- ✅ Custom sign training pipeline

**Tech Stack:** `mediapipe` · `tensorflow` · `pyttsx3` · `sklearn`

---

#### 18. 🌄 [Real-Time Background Replacement](./Real-Time%20Background%20Replacement/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** MediaPipe Selfie Segmentation &nbsp;|&nbsp; **Use Case:** Video Calls, Streaming

Real-time background removal and replacement using MediaPipe's selfie segmentation model. Supports image, video, and blur backgrounds with smooth edge refinement.

**Key Features:**
- ✅ MediaPipe selfie segmentation
- ✅ Background image / video replacement
- ✅ Gaussian blur background mode
- ✅ Edge feathering for clean compositing

**Tech Stack:** `mediapipe` · `opencv` · `numpy`

---

#### 19. 🌀 [Motion-Driven Generative Art Engine](./Motion-Driven%20Generative%20Art%20Engine/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** Optical Flow, Perlin Noise &nbsp;|&nbsp; **Use Case:** Live Performance, Art Installations

Maps human body motion to generative particle systems, flow fields, and Perlin noise canvases. Every unique movement creates a different generative art piece — no two frames are identical.

**Key Features:**
- ✅ Dense optical flow motion mapping
- ✅ Perlin noise-driven flow fields
- ✅ Particle system emitter controls
- ✅ Export final artwork as PNG/MP4

**Tech Stack:** `opencv` · `numpy` · `perlin-noise` · `matplotlib`

---

#### 20. 🧠 [AI Attention & Focus Detector](./AI%20Attention%20%26%20Focus%20Detector/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** MediaPipe Face Mesh, Gaze Estimation &nbsp;|&nbsp; **Use Case:** EdTech, Productivity

Monitors student/worker attention by tracking gaze direction, head pose, and blink rate. Logs attention scores over time and generates focus heatmaps.

**Key Features:**
- ✅ Head-pose estimation (pitch/yaw/roll)
- ✅ Gaze direction estimation
- ✅ Attention score timeline chart
- ✅ Distraction alert with cooldown

**Tech Stack:** `mediapipe` · `opencv` · `matplotlib` · `numpy`

---

#### 21. 🖐️ [Hand-Controlled 3D Object Manipulation](./Hand-Controlled%203D%20Object%20Manipulation/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** MediaPipe Hands, OpenGL, Pygame &nbsp;|&nbsp; **Use Case:** 3D Modeling, Gaming

Manipulate 3D wireframe objects using both hands in real-time. Pinch-to-scale, rotate with wrist rotation, and translate with palm position — no mouse or keyboard needed.

**Key Features:**
- ✅ 6DoF object manipulation
- ✅ Two-hand scale gesture
- ✅ OpenGL 3D rendering
- ✅ Multiple 3D model support

**Tech Stack:** `mediapipe` · `pyopengl` · `pygame` · `numpy`

---

#### 22. 🪞 [AI Emotion Mirror](./AI%20Emotion%20Mirror/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** DeepFace, GAN, OpenCV &nbsp;|&nbsp; **Use Case:** Interactive Art, Wellness

An interactive smart mirror that reflects your emotions back at you with amplified visual effects — happiness adds sparkles, anger adds red distortions, sadness adds rain.

**Key Features:**
- ✅ Real-time emotion recognition
- ✅ Emotion-driven visual overlays
- ✅ Smooth emotion transition blending
- ✅ Mirror flip mode

**Tech Stack:** `deepface` · `opencv` · `numpy`

---

#### 23. 👻 [Time-Echo Camera (Motion Ghosts)](./Time-Echo%20Camera%20%28Motion%20Ghosts%29/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** Frame Buffering, Alpha Blending &nbsp;|&nbsp; **Use Case:** Photography, Art

Simulates long-exposure motion photography by accumulating frame history with fading alpha blending. Moving objects leave beautiful "ghost" trails in their wake.

**Key Features:**
- ✅ Configurable trail length (frame buffer)
- ✅ Alpha fade decay rate control
- ✅ Color echo vs grayscale modes
- ✅ Save echo frames as GIF/video

**Tech Stack:** `opencv` · `numpy` · `collections`

---

#### 24. 🌑 [AI Shadow Art Generator](./AI%20Shadow%20Art%20Generator/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** Background Subtraction, Contours &nbsp;|&nbsp; **Use Case:** Art Installations, Interactive Display

Extracts your shadow (silhouette) in real-time and converts it into artistic representations: chalk drawings, watercolor fills, neon glow outlines, and ink splatter effects.

**Key Features:**
- ✅ MOG2 background subtractor
- ✅ Contour silhouette extraction
- ✅ Multiple artistic style overlays
- ✅ Kaleidoscope shadow symmetry mode

**Tech Stack:** `opencv` · `numpy` · `scipy`

---

#### 25. 🎶 [Motion-to-Music Painter](./Motion-to-Music%20Painter/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** MediaPipe Pose, MIDI, Pygame &nbsp;|&nbsp; **Use Case:** Music, Performance Art

Your body movement becomes a musical instrument and visual painter simultaneously. Arm height controls pitch, lateral movement changes instrument, and speed controls volume.

**Key Features:**
- ✅ Full-body pose landmark tracking
- ✅ Real-time MIDI note generation
- ✅ Motion-to-color painting canvas
- ✅ Multi-instrument pose mapping

**Tech Stack:** `mediapipe` · `mido` · `pygame` · `numpy`

---

#### 26. 🌀 [Reality Distortion Field](./Reality%20Distortion%20Field/)

> **Difficulty:** 🟠 Advanced &nbsp;|&nbsp; **Tech:** Mesh Warping, Remap, Shader Effects &nbsp;|&nbsp; **Use Case:** Art, VFX

Applies mathematically derived real-time distortions to live video: fisheye, swirl, ripple, barrel, and custom warp fields. Distortion parameters are controlled via sliders.

**Key Features:**
- ✅ 8+ distortion effect presets
- ✅ Remap-based warp transforms
- ✅ Interactive live parameter sliders
- ✅ Record distorted video output

**Tech Stack:** `opencv` · `numpy` · `scipy`

---

#### 27. ✨ [AI Aura Visualizer](./AI%20Aura%20Visualizer%20%28energy%20fields%20around%20humans%29/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** MediaPipe Pose, Gaussian Blur, HSV Mapping &nbsp;|&nbsp; **Use Case:** Wellness, Interactive Art

Visualizes a dynamic "energy aura" around the human body by combining pose landmark data with emotion detection, movement velocity, and HSV color mapping into glowing overlays.

**Key Features:**
- ✅ Full-body pose silhouette extraction
- ✅ Emotion-driven aura color mapping
- ✅ Motion velocity → glow intensity
- ✅ Multiple aura style presets

**Tech Stack:** `mediapipe` · `deepface` · `opencv` · `numpy`

---

#### 28. 🖼️ [Living Portrait Generator](./Living%20Portrait%20Generator/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** First Order Motion Model, dlib &nbsp;|&nbsp; **Use Case:** Animation, Art, NFTs

Takes a static portrait photo and animates it with realistic facial movements driven by a live webcam feed — your face becomes the puppet for any portrait.

**Key Features:**
- ✅ First Order Motion Model (FOMM) animation
- ✅ Any portrait photo as source
- ✅ Real-time webcam puppet driving
- ✅ Export animated GIF/MP4

**Tech Stack:** `torch` · `dlib` · `opencv` · `imageio`

---

#### 29. 🔮 [Interactive Kaleidoscope World](./Interactive%20Kaleidoscope%20World/)

> **Difficulty:** 🟡 Intermediate &nbsp;|&nbsp; **Tech:** Affine Transforms, Symmetry Math &nbsp;|&nbsp; **Use Case:** Visual Art, Performance

Transforms your webcam feed into a dynamic, interactive kaleidoscope with configurable symmetry axes, rotation speed, zoom level, and color palette cycling.

**Key Features:**
- ✅ N-fold radial symmetry (4–16 folds)
- ✅ Dynamic rotation speed control
- ✅ Color palette shift over time
- ✅ Mouse/gesture interaction

**Tech Stack:** `opencv` · `numpy` · `math`

---

#### 30. 💭 [AI Dream Camera](./AI%20Dream%20Camera/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** DeepDream, Neural Style Transfer &nbsp;|&nbsp; **Use Case:** Art, AI Research

Applies Google DeepDream and neural style transfer to live video in near-real-time, transforming ordinary webcam footage into psychedelic, dream-like visual hallucinations.

**Key Features:**
- ✅ DeepDream layer-specific activation maximization
- ✅ Neural style transfer (VGG19)
- ✅ Adjustable dream intensity/octaves
- ✅ Multiple dream style presets

**Tech Stack:** `tensorflow` · `torch` · `opencv` · `PIL`

</details>

---

### 🚀 TIER 4 — Flagship & Capstone Projects

<details open>
<summary><b>📁 Click to expand Tier 4 Projects (Flagship)</b></summary>
<br/>

#### 31. ✍️ [ADVANCED 3D AIR DRAWING SYSTEM](./ADVANCED%203D%20AIR%20DRAWING%20SYSTEM/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** MediaPipe, OpenGL, Depth Estimation &nbsp;|&nbsp; **Use Case:** 3D Design, Education

A cutting-edge 3D sketching system that uses hand depth estimation and stereo vision principles to track finger position in X, Y, Z space — enabling true 3D wireframe drawing in the air.

**Key Features:**
- ✅ True 3D finger position tracking (X/Y/Z)
- ✅ OpenGL 3D canvas with orbit camera
- ✅ Depth estimation from hand landmarks
- ✅ Export 3D sketches as .OBJ files
- ✅ Replay mode for playback

**Tech Stack:** `mediapipe` · `pyopengl` · `pygame` · `numpy` · `scipy`

---

#### 32. 🤖 [JARVIS-STYLE 3D HOLOGRAPHIC HUD](./JARVIS-STYLE%203D%20HOLOGRAPHIC%20HUD/)

> **Difficulty:** 🔴 Expert &nbsp;|&nbsp; **Tech:** OpenCV, OpenGL, PyAudio, NLP &nbsp;|&nbsp; **Use Case:** AI Assistant UI, Futuristic Interface

The ultimate capstone project — a full Iron Man JARVIS-inspired 3D holographic heads-up-display with voice commands, real-time system monitoring, rotating arc reactor, animated radar, and holographic overlays.

**Key Features:**
- ✅ Rotating 3D arc reactor animation
- ✅ Animated holographic radar overlay
- ✅ Voice command recognition (speech_recognition)
- ✅ Real-time CPU/RAM/GPU monitoring
- ✅ IP/network info HUD panel
- ✅ Face recognition with name profile
- ✅ Sci-fi typewriter text effects
- ✅ Ambient sound effects & alerts

**Tech Stack:** `opencv` · `pyopengl` · `speechrecognition` · `psutil` · `pyttsx3` · `numpy`

</details>

---

## 🛠️ Technology Stack

<div align="center">

| Category | Technologies |
|:---|:---|
| **Core CV** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EEE?style=flat-square&logo=opencv&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| **Pose / Hands** | ![MediaPipe](https://img.shields.io/badge/MediaPipe-00897B?style=flat-square&logo=google&logoColor=white) ![dlib](https://img.shields.io/badge/dlib-008000?style=flat-square) |
| **Object Detection** | ![YOLO](https://img.shields.io/badge/YOLOv8-FF004F?style=flat-square) ![SSD](https://img.shields.io/badge/SSD-0057e7?style=flat-square) |
| **3D / Graphics** | ![OpenGL](https://img.shields.io/badge/OpenGL-5586A4?style=flat-square&logo=opengl&logoColor=white) ![Pygame](https://img.shields.io/badge/Pygame-3776AB?style=flat-square) |
| **Audio / Speech** | ![PyAudio](https://img.shields.io/badge/PyAudio-333?style=flat-square) ![pyttsx3](https://img.shields.io/badge/pyttsx3-555?style=flat-square) |
| **Emotion / Face AI** | ![DeepFace](https://img.shields.io/badge/DeepFace-00D4FF?style=flat-square) ![FER](https://img.shields.io/badge/FER-E91E63?style=flat-square) |

</div>

---

## ⚙️ Installation

### Prerequisites

```bash
Python 3.8+   |   pip / conda   |   Webcam   |   GPU (recommended for Tier 3-4)
```

### Step 1 — Clone the Repository

```bash
git clone https://github.com/HarshChoudhary2003/Open-CV-Projects.git
cd Open-CV-Projects
```

### Step 2 — Create a Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### Step 3 — Install Core Dependencies

```bash
pip install opencv-python opencv-contrib-python numpy mediapipe tensorflow torch torchvision
pip install face_recognition dlib imutils deepface ultralytics pyautogui
pip install pygame pyttsx3 speechrecognition psutil scipy matplotlib Pillow
```

### Step 4 — Run Any Project

```bash
# Navigate to a project folder
cd "Face Detection + Real-Time Camera"

# Run the main script
python main.py        # or whatever the entry script is named
```

---

## 📁 Project Structure

```
Open-CV-Projects/
│
├── 📁 Face Detection + Real-Time Camera/
├── 📁 Face Mask Detection/
├── 📁 AI Virtual Mouse/
├── 📁 Face Recognition Attendance System/
├── 📁 Invisible UI Controller/
├── 📁 Driver Drowsiness & Distraction Detection System/
├── 📁 Object_Detection_YOLO/
├── 📁 Air Drawing App/
├── 📁 Motion-Based Light Painting/
├── 📁 Gesture-Based Mandala -- Generative Art/
├── 📁 Handwritten Digit Drawing --AI Prediction/
├── 📁 AR Face Filters/
├── 📁 Emotion-Based Color Painting/
├── 📁 Creative Computer Vision Studio (All-in-One App)/
├── 📁 Gesture-Controlled Music Visualizer/
├── 📁 AR Try-On System/
├── 📁 AI Sign Language Translator/
├── 📁 Real-Time Background Replacement/
├── 📁 Motion-Driven Generative Art Engine/
├── 📁 AI Attention & Focus Detector/
├── 📁 Hand-Controlled 3D Object Manipulation/
├── 📁 AI Emotion Mirror/
├── 📁 Time-Echo Camera (Motion Ghosts)/
├── 📁 AI Shadow Art Generator/
├── 📁 Motion-to-Music Painter/
├── 📁 Reality Distortion Field/
├── 📁 AI Aura Visualizer (energy fields around humans)/
├── 📁 Living Portrait Generator/
├── 📁 Interactive Kaleidoscope World/
├── 📁 AI Dream Camera/
├── 📁 ADVANCED 3D AIR DRAWING SYSTEM/
├── 📁 JARVIS-STYLE 3D HOLOGRAPHIC HUD/
│
└── 📄 README.md
```

---

## 🗺️ Project Roadmap

```
[v1.0] ──► Basic Detection (Projects 1–8)
    │
[v2.0] ──► Interactive & Creative (Projects 9–16)
    │
[v3.0] ──► AI-Powered Systems (Projects 17–26)
    │
[v4.0] ──► Expert / Flagship Systems (Projects 27–32)
    │
[v5.0] ──► 🔜 WebRTC Integration, Mobile, Cloud Deployment
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. 🍴 Fork this repository
2. 🌿 Create a feature branch: `git checkout -b feature/my-new-project`
3. 💾 Commit your changes: `git commit -m "feat: add my new CV project"`
4. 📤 Push to the branch: `git push origin feature/my-new-project`
5. 🔃 Open a Pull Request

Please ensure your project:
- ✅ Has a clear `README.md` inside the project folder
- ✅ Lists all dependencies in `requirements.txt`
- ✅ Includes at least one demo screenshot or GIF
- ✅ Follows the existing folder naming convention

---

## 📝 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

## 👤 Author

<div align="center">

<img src="https://github.com/HarshChoudhary2003.png" width="120px" style="border-radius:50%"/>

### **Harsh Choudhary**

*Computer Vision Engineer | AI Developer | Open Source Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-@HarshChoudhary2003-181717?style=for-the-badge&logo=github)](https://github.com/HarshChoudhary2003)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Harsh%20Choudhary-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/harshchoudhary2003)
[![Email](https://img.shields.io/badge/Email-hc504360@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:hc504360@gmail.com)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,50:00D4FF,100:00FF88&height=120&section=footer" width="100%"/>

**⭐ Star this repository if you found it useful!**

*Built with ❤️ and lots of ☕ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)*

⬆️ [Back to Top](#-open-cv-projects)

</div>
