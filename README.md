### 🌾 Noise-Reduction-Oriented Super-Resolution Reconstruction for Precision Agriculture Applications

This repository accompanies the project:

> **"Noise-Reduction-Oriented Super-Resolution Reconstruction for Precision Agriculture Applications"**

The goal of this work is to enhance image quality in precision agriculture by leveraging super-resolution reconstruction with an emphasis on noise reduction. The approach aims to recover fine spatial details while suppressing noise to improve analysis accuracy for agricultural monitoring and decision support.

---

## 📂 Repository Structure

- 🏋️ **training/** — Contains scripts and configuration files for model training.  

- 🔍 **inference/** — Contains inference and evaluation scripts.

Pretrained checkpoints can be downloaded directly from this folder or from the link below.

🧠 Pretrained Models and Checkpoints

Download pretrained super-resolution models from:

[👉 Download Checkpoints](https://drive.google.com/drive/folders/1dLSxTwEyA8oP7H10xSxn_U3P_h33qQli?usp=sharing)

After downloading, place the files as follows:

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

 ```bash
pip install -r requirements.txt
 ```

🚀 How to Run

✅ 1. Train the Model

Run the training script:

 ```bash
cd training
bash run.sh
 ```

🔬 2. Inference and Evaluation

Navigate to the inference folder and execute:

 ```bash
cd inference
bash run_SR.sh
 ```

📧 Contact

For any questions, issues, or collaboration inquiries, please contact:

Minh Trieu Tran

📨 minhtrieu.tran@gssi.it
