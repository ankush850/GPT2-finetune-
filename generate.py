import argparse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="finetuned_model", help="Path to fine-tuned model folder or HF model id")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=120)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (more creative)")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    outputs = gen(
        args.prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        do_sample=args.do_sample or True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.eos_token_id
    )

    for i, out in enumerate(outputs, 1):
        print(f"\n=== Generated #{i} ===\n")
        print(out["generated_text"])

if __name__ == "__main__":
    main()
