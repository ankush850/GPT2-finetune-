import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import datetime

# -------------------------------
# Load Fine-tuned GPT-2 Model
# -------------------------------
MODEL_PATH = "finetuned_model"

print(f"[{datetime.datetime.now()}] Loading tokenizer and model from: {MODEL_PATH}")

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# -------------------------------
# Text Generation Function
# -------------------------------
def generate_text(prompt, max_length, temperature, top_k, top_p, num_return_sequences):
    """
    Generate text from the fine-tuned GPT-2 model with customizable decoding parameters.
    """
    if not prompt.strip():
        return "⚠️ Please enter a valid prompt!"

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    results = []
    for i in range(num_return_sequences):
        text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        results.append(f"**Generated {i+1}:**\n{text}\n")

    return "\n\n---\n\n".join(results)


# -------------------------------
# Helper for Model Info
# -------------------------------
def get_model_info():
    info = f"""
    **Model Path:** {MODEL_PATH}  
    **Device Used:** {device}  
    **Model Type:** GPT-2 (fine-tuned)  
    **Torch Version:** {torch.__version__}  
    **Transformers Version:** 4.x  
    """
    return info


# -------------------------------
# Gradio Interface
# -------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 Fine-Tuned GPT-2 Text Generator
        Generate creative text outputs from your **fine-tuned GPT-2 model**.  
        Customize parameters like **temperature, top-k, top-p, and max length** for more control.  
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="📝 Enter your prompt",
                placeholder="Type something like 'Once upon a time in a futuristic world...'",
                lines=4
            )
            generate_btn = gr.Button("✨ Generate Text")

        with gr.Column(scale=2):
            max_length = gr.Slider(20, 500, value=150, step=10, label="Max Length")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
            top_k = gr.Slider(0, 100, value=50, step=5, label="Top-k")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            num_return_sequences = gr.Slider(1, 5, value=1, step=1, label="Number of Responses")

    with gr.Row():
        output = gr.Markdown(label="Generated Output")

    with gr.Accordion("ℹ️ Model Info", open=False):
        gr.Markdown(get_model_info())

    with gr.Accordion("📌 Example Prompts", open=False):
        examples = [
            ["Once upon a time in India"],
            ["The future of Artificial Intelligence is"],
            ["In a world where humans and robots coexist"],
            ["A mysterious treasure was hidden inside the castle"],
            ["Write a motivational speech about success"]
        ]
        gr.Examples(examples=examples, inputs=prompt)

    # Bind button
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt, max_length, temperature, top_k, top_p, num_return_sequences],
        outputs=output
    )


# -------------------------------
# Launch Gradio App
# -------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
