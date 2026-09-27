
# 🌲 Real-Time Computer Vision Game Automation

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, purely vision-based automation pipeline designed to play rapid-response arcade games. By leveraging real-time screen capturing and advanced image processing algorithms, this system autonomously perceives the game environment, identifies dynamic obstacles, and calculates safe movement trajectories without interacting with the game's internal memory or API.

## 📖 System Architecture & Computer Vision Pipeline

The bot operates on a continuous loop, executing a precise computer vision pipeline on every single frame captured from the designated gameplay window.

1. **Ultra-Low Latency Frame Capture:** Utilizes the `mss` library to grab specific screen coordinates (Regions of Interest) at high frame rates, eliminating the input lag commonly associated with standard screenshot modules.
2. **Color Space Segmentation:** Converts raw BGR frames to the **HSV color space** to create precise binary masks. This isolates critical game entities (the player character, the tree trunk, branches, and the energy bar) regardless of minor lighting or background variations.
3. **Morphological Noise Reduction:** Applies dilation and closing operations to the binary masks to fill gaps and remove visual noise, ensuring solid, contiguous shapes for contour detection.
4. **Geometric Centroid Calculation:** Extracts contours and calculates spatial moments (`cv2.moments`) to pinpoint the exact $x, y$ center coordinates of the player and the tree trunk.
5. **Dynamic ROI Collision Detection:** Generates custom polygonal scanning zones (`cv2.polylines`) on the immediate left and right flanks of the tree trunk. The system analyzes these zones for intersecting branch pixels to predict imminent collisions.
6. **Hough Transform Line Detection:** Implements `cv2.HoughLines` to mathematically verify the vertical boundaries of the tree trunk, ensuring the bot's spatial awareness remains calibrated even during fast scrolling.

## 🛠️ Technology Stack

| Component | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Core Logic** | `Python 3.x` | Primary programming language handling the operational loop. |
| **Computer Vision** | `OpenCV (cv2)` | Image processing, masking, morphological operations, and geometry. |
| **Matrix Operations** | `NumPy` | Fast array manipulations required for image masking and masking arithmetic. |
| **Screen Capture** | `mss` | Real-time, thread-safe screen grabbing tailored for high FPS. |
| **System I/O** | `time`, `os` | Loop timing, delay management, and environment checks. |

## ⚙️ Installation & Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/AUBAI-ALKHABBAZ/Automating-game-using-OpenCV.git](https://github.com/AUBAI-ALKHABBAZ/Automating-game-using-OpenCV.git)
cd Automating-game-using-OpenCV
