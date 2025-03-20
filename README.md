🏆 YOLO Object Detection with COCO2017 Subset

🌍 Vision

To develop an inclusive and efficient object detection system that enhances AI-powered recognition across various industries such as security, surveillance, autonomous vehicles, and smart cities.

🎯 Mission

Our mission is to build an accurate, real-time, and user-friendly object detection model using state-of-the-art deep learning techniques and provide easy deployment for real-world applications.

🚀 Features

✅ YOLOvX Model fine-tuned for improved accuracy🎥 Real-Time Object Detection for images and videos⚡ High-Speed Performance optimized for fast inference📊 Custom Dataset Training using the COCO2017 subset🌐 Streamlit Web Interface for interactive user experience📈 Supports Edge & Cloud Deployment for flexibility

🔍 Technologies Used

Python: Core development language

YOLOvX: Pretrained model for object detection

OpenCV: Image processing and visualization

Streamlit: Web-based UI for real-time interaction

COCO2017 Dataset: Training and validation dataset

PyTorch: Deep learning framework

📂 Project Structure

📜 Dataset & Conversion to YOLO Format

We used the COCO2017 subset, which contains images and annotations in JSON format. To convert it into YOLO format, we performed:

1️⃣ Dataset Download: Extract relevant classes for training.2️⃣ Annotation Conversion: Convert COCO JSON to YOLO format (class_id, x_center, y_center, width, height).3️⃣ Data Organization: Store images in datasets/images/ and labels in datasets/labels/.4️⃣ Verification: Use OpenCV and label visualization tools to ensure correctness.

⚙️ Installation & Setup

Clone the repository and install dependencies:

🚀 Usage

🎯 Run Object Detection

Run inference on an image:

Run inference on a video:

🌐 Launch Streamlit Web App

To provide a real-time interactive UI, run:

This will launch a web application where users can upload images/videos and visualize detection results in real-time.

📊 Model Training Details

Model: YOLOvX

Dataset: COCO2017 Subset

Epochs: XX

Batch Size: XX

Optimizer: Adam/SGD

Augmentations: Applied various data augmentation techniques

📈 Performance Metrics

📊 mAP@50: XX%⏩ Inference Speed: XX FPS🎯 Precision & Recall: XX / XX

🎨 Sample Results

🔥 Future Enhancements

🔹 Expand dataset for better generalization🔹 Optimize model for mobile & edge devices🔹 Implement real-time multi-object tracking🔹 Enhance Streamlit UI with better animations & interactivity

💡 Contributing

We welcome contributions! If you’d like to improve the project, please fork the repo, make changes, and submit a pull request.

🙏 Acknowledgments

Special thanks to [Your Mentor's Name] for guidance and the Ultralytics YOLO community for their contributions.
