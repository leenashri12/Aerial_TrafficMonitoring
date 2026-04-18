# Aerial Traffic Monitoring & Analysis

A specialized tool developed to track and analyze vehicle flow from drone-captured footage. By combining custom-trained YOLO models with object tracking, the system provides real-time data on traffic volume and congestion levels.

###  Core Features
* **Persistent Tracking:** Uses ByteTrack to assign unique IDs to vehicles, ensuring accurate counts without duplicates.
* **Flexible Tripwires:** Supports both vertical and horizontal counting lines to match any road orientation or camera angle.
* **Live Analytics:** Calculates real-time flow rate (vehicles per second) and labels congestion as Low, Medium, or High.
* **Aerial Optimized:** Fine-tuned on the VisDrone dataset to maintain high accuracy for small objects and high-altitude views.

###  Tech Stack
* **Model:** YOLO (Custom trained on VisDrone via Roboflow)
* **Tracking:** ByteTrack + Supervision
* **Interface:** Streamlit
* **Training:** Kaggle (Tesla P100 GPU)

###  Project Structure
* `app.py`: Main application script.
* `weights/best.pt`: The custom-trained model weights.
* `test_assets/`: Folder containing sample videos and images for demonstration.
* `requirements.txt`: Necessary Python libraries.

###  How to Run
1.  **Install dependencies:** `pip install -r requirements.txt`
2.  **Launch the dashboard:** `streamlit run app.py`

###  Development Journey
* **Training:** Processed 50 epochs on a Tesla P100 GPU. Implemented resume logic to ensure full convergence, achieving a 11.4ms inference speed.
* **Logic:** Built custom spatial triggers to detect directional flow (In/Out) and integrated temporal math for flow rate calculations.
* **Problem Solving:** Debugged Windows-specific file access issues (WinError 32) and optimized the video processing loop for smoother playback.
* **Testing:** Validated the system across diverse environmental scenarios, including night-time footage and high-density urban intersections.