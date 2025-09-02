import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load Model
MODEL_PATH = "finetuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Text Generation
def generate_text(prompt, max_length, temperature, top_k, top_p, num_return_sequences):
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
    return "\n\n---\n\n".join(
        [f"**Output {i+1}:**\n{tokenizer.decode(outputs[i], skip_special_tokens=True)}"
         for i in range(num_return_sequences)]
    )

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Fine-Tuned GPT-2 Generator")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=3)
        generate_btn = gr.Button("✨ Generate")
    with gr.Row():
        max_length = gr.Slider(20, 300, 150, step=10, label="Max Length")
        temperature = gr.Slider(0.1, 1.5, 0.8, step=0.1, label="Temperature")
        top_k = gr.Slider(0, 100, 50, step=5, label="Top-k")
        top_p = gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Top-p")
        num_return_sequences = gr.Slider(1, 3, 1, step=1, label="Responses")
    output = gr.Markdown()

    gr.Examples(
        [["Once upon a time"], ["AI will change the world"], ["A treasure was hidden"]],
        inputs=prompt
    )

    generate_btn.click(
        fn=generate_text,
        inputs=[prompt, max_length, temperature, top_k, top_p, num_return_sequences],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)
