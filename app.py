import gradio as gr from transformers import GPT2LMHeadModel, GPT2Tokenizer # Load fine-tuned model model_path = "finetuned_model" tokenizer = GPT2Tokenizer.from_pretrained(model_path) model = GPT2LMHeadModel.from_pretrained(model_path) def generate_text(prompt): inputs = tokenizer.encode(prompt, return_tensors="pt") outputs = model.generate( inputs, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95 ) return tokenizer.decode(outputs[0], skip_special_tokens=True) # Gradio interface demo = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="Fine-Tuned GPT-2", description="Apna khud ka trained GPT-2 text generator 🚀") demo.launch()import gradio as gr from transformers import GPT2LMHeadModel, GPT2Tokenizer # Load fine-tuned model model_path = "finetuned_model" tokenizer = GPT2Tokenizer.from_pretrained(model_path) model = GPT2LMHeadModel.from_pretrained(model_path) def generate_text(prompt): inputs = tokenizer.encode(prompt, return_tensors="pt") outputs = model.generate( inputs, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95 ) return tokenizer.decode(outputs[0], skip_special_tokens=True) # Gradio interface demo = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="Fine-Tuned GPT-2", description="Apna khud ka trained GPT-2 text generator 🚀") demo.launch()import gradio as gr from transformers import GPT2LMHeadModel, GPT2Tokenizer # Load fine-tuned model model_path = "finetuned_model" tokenizer = GPT2Tokenizer.from_pretrained(model_path) model = GPT2LMHeadModel.from_pretrained(model_path) def generate_text(prompt): inputs = tokenizer.encode(prompt, return_tensors="pt") outputs = model.generate( inputs, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95 ) return tokenizer.decode(outputs[0], skip_special_tokens=True) # Gradio interface demo = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="Fine-Tuned GPT-2", description="Apna khud ka trained GPT-2 text generator 🚀") demo.launch()import gradio as gr from transformers import GPT2LMHeadModel, GPT2Tokenizer # Load fine-tuned model model_path = "finetuned_model" tokenizer = GPT2Tokenizer.from_pretrained(model_path) model = GPT2LMHeadModel.from_pretrained(model_path) def generate_text(prompt): inputs = tokenizer.encode(prompt, return_tensors="pt") outputs = model.generate( inputs, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95 ) return tokenizer.decode(outputs[0], skip_special_tokens=True) # Gradio interface demo = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="Fine-Tuned GPT-2", description="Apna khud ka trained GPT-2 text generator 🚀") demo.launch()import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model
model_path = "finetuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio interface
demo = gr.Interface(fn=generate_text,
                    inputs="text",
                    outputs="text",
                    title="Fine-Tuned GPT-2",
                    description="Apna khud ka trained GPT-2 text generator 🚀")

demo.launch()
