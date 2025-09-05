GPT-2 Fine-Tuning Web App 🚀

This project is a Flask-based web application for fine-tuning OpenAI’s GPT-2 model on custom datasets and generating text directly from a browser frontend.

📂 Project Structure
GPT2-finetune-/
│── data/                # Place your dataset here (mydata.txt)
│── finetuned_model/     # Saved fine-tuned model
│── static/              # CSS, JS, images for frontend
│── templates/           # HTML templates (Flask frontend)
│── logs/                # Training logs (TensorBoard)
│── train.py             # Script to fine-tune GPT-2 (can be triggered via Flask)
│── generate.py          # Script for text generation
│── app.py               # Flask backend (main entry point)
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation




⚡ Features

Fine-tune GPT-2 on your own text dataset.

Simple frontend UI to upload dataset, train, and generate text.

Supports GPU acceleration if CUDA is available.

TensorBoard integration for training monitoring.

Clean Flask architecture with templates/ (HTML) + static/ (CSS/JS).






🛠️ Installation

Clone the repository

git clone https://github.com/ankush850/GPT2-finetune-.git
cd GPT2-finetune-


Create & activate virtual environment

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt

📑 Preparing Your Dataset

Place your text dataset at data/mydata.txt.

Format: each line = one training example (sentence/paragraph).

Example:

The sun sets behind the mountains.
AI is transforming industries.
Once upon a time, there was a brave knight...





🚀 Running the Web App

Start the Flask app:

python app.py

Now open in browser: http://127.0.0.1:5000





🎯 Usage (Frontend)

Upload Dataset → Provide your .txt file for training.

Train Model → Start fine-tuning directly from the web UI.

Generate Text → Enter a prompt and get AI-generated text.
