Tomato Leaf Disease Classification using CNN
This project implements a Deep Learning pipeline to classify diseases in tomato leaves using the PlantVillage dataset. It includes a trained Convolutional Neural Network (CNN) and a Flask-based REST API to serve predictions.

📌 Project Overview

    The goal is to provide an automated way to detect plant pathology via images. The model identifies several categories, including:
    Bacterial Spot
    Early Blight
    Late Blight
    Leaf Mold
    Septoria Leaf Spot
    Spider Mites (Two-spotted spider mite)
    Yellow Leaf Curl Virus
    Healthy

🏗️ Architecture
	The system consists of two main components:
	Model Training: A CNN architecture built with TensorFlow/Keras, utilizing layers like Conv2D, MaxPooling2D, Dropout, and Dense.
	Flask API: A lightweight web server that accepts image uploads and returns JSON predictions.

🚀 Getting Started

1. Prerequisites
   	Ensure you have Python 3.11+ installed.
   	It is recommended to use a virtual environment.
   	Bash python -m venv venv
   	source venv/bin/activate
   	# On Windows: venv\Scripts\activate
2. Installation
   	Clone the repository and install the required dependencies:
  	 Bash
  		git clone https://github.com/<username>/<repo>>.git
   		cd <root_folder>
   		pip install -r requirements.txt
3. Dataset
   	Download the Tomato dataset from Kaggle (PlantVillage). Place the images in the data/ directory.

🧠 Model Training
	To train the model from scratch, run the training script. This will perform data augmentation, training, and save the final model as an .keras file.
	Bash 
		python train.py
	
	Input Size: 256x256 (default)
	Optimizer: Adam
	Loss Function: Sparse Categorical Crossentropy

🌐 API Deployment
	The Flask API allows other applications to send images for classification.
	Start the server:
	Bash
		python main.py
	The server will start at http://127.0.0.1:5000.

    API Endpoint:
        URL: /predict
        Method: POST
        Payload: Form-data with key file (the image)
        Example Response:
        JSON{
                "class": "Tomato_Early_blight",
                "confidence": 0.984
            }

📊 Performance
	Metric Value
	Training Accuracy ~96%
	Validation Accuracy ~92%
	Epochs 25

📂 Project Structure
	Plaintext
	├── data/ # Raw and processed images
	├── models/ # Saved .h5 model files
	├── train/ # Exploration and prototyping
	├── flaskapi # Flask API
	└── requirements.txt # Project dependencies
