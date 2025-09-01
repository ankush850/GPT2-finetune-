# GPT-2 Fine-Tune (VS Code Project)

Train and use a GPT-2 text-generation model on **your own dataset** directly in VS Code.

## 🚀 Quick Start

1) **Create/activate venv**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
```

3) **Put your dataset**
- Add a plain text file at: `data/mydata.txt`
- Each line can be a sample paragraph, sentence, etc. The model learns your style.

4) **Train**
```bash
python train.py
```
- Fine-tuned model is saved to `finetuned_model/`

5) **Generate**
```bash
python generate.py --prompt "Write a short poem about rain" --max_length 120
```

## ⚙️ Notes / Tips
- Start with the default `gpt2` model for low VRAM machines. You can change to `gpt2-medium` via `--model_name`.
- If you have an NVIDIA GPU with CUDA installed, PyTorch will automatically use it.
- For larger datasets, raise `--num_train_epochs` and `--block_size` as needed.
- Logs go to `./logs`; view in TensorBoard:
  ```bash
  tensorboard --logdir logs
  ```

## 🧪 Example dataset
A tiny example is provided at `data/mydata.txt`. Replace with your data for real results.
