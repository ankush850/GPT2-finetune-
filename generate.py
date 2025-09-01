import argparse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT-2 model or a Hugging Face model ID.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="finetuned_model",
        help="Path to the fine-tuned model directory or Hugging Face model identifier."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Initial text prompt to start the generation from."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=120,
        help="Maximum length of the generated sequence including the prompt."
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of generated sequences to return."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Higher values mean more random generations."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability threshold."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling for generation instead of greedy decoding."
    )
    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    """
    Load the tokenizer and model from the specified path or model ID.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer, model

def create_text_generation_pipeline(model, tokenizer):
    """
    Create a Hugging Face pipeline for text generation.
    """
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_text(
    generator,
    prompt,
    max_length,
    num_return_sequences,
    do_sample,
    temperature,
    top_p,
    pad_token_id
):
    """
    Generate text sequences using the pipeline with the specified parameters.
    """
    outputs = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample or True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_token_id
    )
    return outputs

def print_generated_texts(outputs):
    """
    Print the generated text sequences to the console.
    """
    for i, output in enumerate(outputs, start=1):
        print(f"\n=== Generated Sequence #{i} ===\n")
        print(output["generated_text"])

def main():
    args = parse_arguments()

    tokenizer, model = load_model_and_tokenizer(args.model_path)

    generator = create_text_generation_pipeline(model, tokenizer)

    outputs = generate_text(
        generator=generator,
        prompt=args.prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.eos_token_id
    )

    print_generated_texts(outputs)

if __name__ == "__main__":
    main()