### 🌾 Noise-Reduction-Oriented Super-Resolution Reconstruction for Precision Agriculture Applications

This repository accompanies the project:

> **"Noise-Reduction-Oriented Super-Resolution Reconstruction for Precision Agriculture Applications"**

The goal of this work is to enhance image quality in precision agriculture by leveraging super-resolution reconstruction with an emphasis on noise reduction. The approach aims to recover fine spatial details while suppressing noise to improve analysis accuracy for agricultural monitoring and decision support.

---

## 📂 Repository Structure

- 🏋️ **training/** — Contains scripts and configuration files for model training.  
  To start training, run the bash script:
  ```bash
  cd training
  bash run_train.sh
🔍 inference/ — Contains inference and evaluation scripts.
Includes a log/ folder with reported experimental results.
Pretrained checkpoints can be downloaded directly from this folder or from the link below.

🧠 Pretrained Models and Checkpoints
Download pretrained super-resolution models from:
👉 Download Checkpoints

After downloading, place the files as follows:

arduino
Copy code
project-root/
├── training/
│   └── run.sh
├── inference/
│   ├── log/
│   └── [downloaded_checkpoints_here]
🧬 Dataset Information
🖼️ Training Dataset
DIV2K — A high-quality benchmark for image super-resolution.
Download from:
🔗 https://data.vision.ee.ethz.ch/cvl/DIV2K/

🎯 Saliency Maps and 🧪 Test Datasets
We generate custom saliency maps by applying Gaussian filtering to enhance spatial attention.
Download saliency data and test datasets here:
👉 https://drive.google.com/drive/folders/1_X74k8hHqH-r0lA9RR5hrm4PKX3sI6rc?usp=sharing

⚙️ Environment Setup
We recommend using a Python virtual environment (conda or venv).

🐍 Required Dependencies
Install all dependencies via:

bash
Copy code
pip install -r requirements.txt
🚀 How to Run
✅ 1. Train the Model
Run the training script:

bash
Copy code
cd training
bash run.sh
🔬 2. Inference and Evaluation
Navigate to the inference folder and execute:

bash
Copy code
cd inference
bash run_SR.sh
📧 Contact
For any questions, issues, or collaboration inquiries, please contact:

Minh Trieu Tran
📨 minhtrieu.tran@gssi.it
