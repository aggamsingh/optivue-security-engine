# 🚨 Optivue: Multi-Modal Edge-AI Security & Behavioral Analytics Engine

### **Project Overview**
Optivue is an integrated, production-grade computer vision system engineered to provide automated threat verification, hazard detection, and behavioral analytics. By orchestrating four parallel deep learning and geometric inference layers, the engine elevates standard security streams from basic pixel tracking to contextual, high-level situational awareness. The entire architecture is optimized for low-latency hardware acceleration and features an HTML5 WebRTC live-streaming browser frontend.

---

## 🏗️ Core System Architecture
```text
┌────────────────────────┐
              │ 📷 Live Webcam Stream  │
              └───────────┬────────────┘
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
┌───────────────────────┐     ┌───────────────────────┐
│ 🔥 Custom YOLOv8m     │     │ 👥 Standard YOLOv8n   │
│  (Fire/Smoke Tracking)│     │  (Human Isolation)    │
└───────────────────────┐     └───────────┬───────────┘
│
▼
┌───────────────────────┐
│ 🤸‍♂️ YOLOv8-Pose        │
│  (17 Joint Keypoints) │
└───────────┬───────────┘
│
▼
┌───────────────────────┐
│ 🔐 face_recognition   │
│  (128-D Vector Match) │
└───────────┬───────────┘
│
┌──────────────┴──────────────┐
▼                             ▼
┌───────────────────────┐     ┌───────────────────────┐
│ ✅ AUTHORIZED USER    │     │ ❌ CRITICAL INTRUDER  │
└───────────────────────┘     └───────────────────────┘

## 📊 Dataset & Training Methodology

### **Module 2: Custom Fire & Smoke Dataset**
* **Dataset Composition:** Curated custom multi-class image set specifically balancing high-variance outdoor, industrial, and residential environments containing visible flames and varying densities of smoke.
* **Data Augmentation:** Bounding boxes were regularized against variations in environmental lighting and camera motion artifacts via random horizontal flipping, exposure adjustments, scaling, and HSV color-space modifications.
* **Annotation Protocol:** Multi-class labels mapped to specialized class IDs:
    * `Class 0: FIRE` (Tight-boundary flame coordinates)
    * `Class 1: SMOKE` (Volumetric bounding region clustering)

### **Hyperparameter & Training Setup**
* **Base Framework:** Ultralytics YOLOv8 Medium (`yolov8m.pt`) balanced for spatial optimization and feature representation complexity.
* **Hardware Accelerator:** NVIDIA T4 Tensor Core GPU (CUDA execution context).
* **Training Duration:** 50 Full Epochs.
* **Optimization Function:** Stochastic Gradient Descent (SGD) utilizing dynamic learning rate scheduling (`lr0=0.01`, `lrf=0.01`) and standard AdamW weight decay regularizations to avoid over-fitting on background non-hazard clusters.
* **Loss Metrics:** Tracked via Box Loss (CIoU) and Class Loss (BCE) to guarantee localization stability.

---

## 🧠 Multi-Layered Technical Modules

### **Module 1: Real-Time Human Detection & Isolation**
* **Technology:** YOLOv8 Nano (`yolov8n.pt`) lightweight object detector.
* **Execution Role:** Acts as the primary spatial filter. The framework intercepts incoming frame matrices, targets the coco class index `0 (person)`, and slices the bounding array coordinates out of the global pixel matrix at a strict confidence threshold ($>0.40$).

### **Module 2: Custom Hazard Tracking (Fire & Smoke)**
* **Technology:** Fine-Tuned Custom YOLOv8m Model weights (`best.pt`).
* **Execution Role:** Scans ambient conditions independently of human targets. If pixel matrices match custom trained features, it overlays dynamic alert bounding boxes over active flame boundaries (crimson red) and billowing smoke clusters (orange warning).

### **Module 3: 128-Dimensional Facial Whitelisting Interface**
* **Technology:** `face_recognition` API supported by a pre-trained **dlib ResNet-13** facial feature backbone network.
* **Execution Role:** When Module 1 isolates a human figure, Module 3 automatically maps the facial bounding box region. It computes facial landmarks to extract a highly precise **128-dimensional floating-point vector embedding**.
* **Verification Math:** The live vector is cross-examined against a localized, securely cached reference database vector via **Euclidean Distance Evaluation**:

$$\text{Distance} = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$

* If $\text{Distance} \le 0.55$, a definitive face vector match overrides generic human detection to frame the user in a **Safe Green Banner** (`AUTHORIZED USER`).
* If $\text{Distance} > 0.55$, the engine immediately escalates safety flags to render a **Crimson Alert Banner** (`CRITICAL INTRUDER`).

### **Module 4: Behavioral Posture & Fall Detection Logic**
* **Technology:** YOLOv8-Pose Estimation network (`yolov8n-pose.pt`).
* **Execution Role:** Extracts the human structural skeleton by tracking **17 specific bio-mechanical keypoint nodes** (shoulders, hips, knees, ankles, etc.) inside the image canvas matrix.

![YOLOv8 Pose Skeleton Keypoint Mapping](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolov8-pose-keypoints.png)

* **Fail-Safe Occlusion Geometry:** To handle real-world challenges where objects (like desks or plants) hide lower limbs, the module executes a two-layered mathematical safety net:
    1.  **Skeletal Vector Delta Check:** If both left/right shoulder (Indices 5, 6) and hip (Indices 11, 12) nodes output confidence scores $>0.50$, the script calculates the absolute vertical pixel height of the torso. If this spatial height collapses down to less than 25% of the total bounding box height ($y_{\text{delta}} < h \times 0.25$), a fall alert triggers.
    2.  **Bounding Box Aspect-Ratio Override:** If body parts are completely hidden or occluded, the keypoint metrics fail. The model automatically falls back to standard bounding box aspect ratios ($\text{Width} / \text{Height}$). If the bounding box width shifts to scale wider than its height by a factor of 1.2 ($\text{AR} > 1.2$), a horizontal structural orientation is verified and forces a `CRITICAL FALL DETECTED` state.

---

## 🎨 Web Frontend & Live Pipeline Execution

* **UI Delivery:** Designed as a seamless browser application via inline asynchronous cells inside Google Colab environments.
* **Webcam Optimization:** Standard API streaming routes frames via cloud detours, which introduces heavy lag. To bypass this bottleneck, Optivue implements a native browser **HTML5 JavaScript WebRTC / MediaStream interface layer** mapped inside the Google Colab execution canvas.
* **The Execution Loop:**
    1.  JavaScript grabs raw camera pixels directly from laptop hardware, renders them instantly to a localized `<video>` element, and passes the array out as a compressed JPEG **Base64 string sequence** over an active asynchronous socket link.
    2.  Python catches the Base64 stream, decodes it into a writable **NumPy matrix format**, maps CUDA tensors, and distributes the frame simultaneously through the 4 AI backends.
    3.  OpenCV paints real-time status banners on the canvas array before it is re-encoded to a Base64 string and pushed straight back to an interactive browser `<canvas>` tag at **10 frames per second** with virtually zero input lag.

---

## 🛠️ Complete Tech Stack Listing

* **Programming Language:** Python 3.12+
* **Deep Learning Frameworks:** PyTorch (CUDA Accelerated Engine Core), Ultralytics (YOLOv8 Suite)
* **Computer Vision Libraries:** OpenCV (`cv2`), `face_recognition` (dlib feature extraction models), Pillow (`PIL`), NumPy
* **Web Integration Tools:** Google Colab Async `eval_js` API, HTML5 Canvas, WebRTC MediaStream JavaScript API Frameworks
* **Storage Framework:** Persistent Cloud-Mounted Filesystems (Google Drive Interoperability Hooks)
