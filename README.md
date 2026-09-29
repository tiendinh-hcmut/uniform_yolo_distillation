# Uniform YOLO Distillation

An end-to-end knowledge distillation pipeline designed to automate the labeling and detection of specific uniform items (e.g., green shirts, blue shirts) in real-time video streams. 

This project utilizes large, zero-shot Vision Foundation Models as "Teacher" networks to auto-annotate raw datasets, which are then used to train lightweight YOLO "Student" models for high-speed, real-time inference.

## 🚀 Pipeline Workflow
1. **Teacher Inference & Auto-labeling:** Utilize zero-shot object detection models (YOLO-World, Grounding DINO) to scan raw video frames and automatically generate bounding box annotations for target classes.
2. **Dataset Formatting:** Format the generated labels into YOLO standard structure (80/20 train/val split).
3. **Student Training:** Train a lightweight Student model (YOLO26n) on the auto-generated dataset to learn the Teacher's knowledge.
4. **Real-time Video Inference:** Deploy the trained Student model on video streams using OpenCV to detect target uniform classes in real-time.

## 📂 Repository Structure
* `/yoloworld_distillation/`: Contains the pipeline for auto-labeling with YOLO-World and training the student YOLO model.
* `/groundingdino_distillation/`: Alternative distillation pipeline leveraging Grounding DINO for complex zero-shot text-to-box labeling.
* `extract_frames.py`: Script to extract frame images from raw uniform videos.
* `download_youtubevid.py`: Utility to fetch target videos for testing and dataset creation.
* `split_dataset.py`: Utility to randomly split raw images and labels into standard YOLO train/val formats.

## 🛠️ Setup & Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/tiendinh-hcmut/uniform_yolo_distillation.git](https://github.com/tiendinh-hcmut/uniform_yolo_distillation.git)
   cd uniform_yolo_distillation
   ```
2. Activate your virtual environment and install dependencies:
   ```bash
   pip install ultralytics opencv-python pillow
   ```

## 🎯 Usage
Navigate to the desired distillation folder (e.g., `yoloworld_distillation`) and execute the pipeline steps in order:

**1. Auto-label the dataset:**
```bash
python 1_auto_label.py
```

**2. Train the Student Model:**
```bash
python 2_train_student.py
```

**3. Run Real-time Video Inference:**
```bash
python 3_video_demo.py
```