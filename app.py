from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize Flask application instance
flask_application_instance = Flask(__name__)

# Define the path to the fine-tuned GPT-2 model directory
fine_tuned_model_directory_path = "finetuned_model"

# Load the tokenizer from the specified model directory path
tokenizer_instance = GPT2Tokenizer.from_pretrained(fine_tuned_model_directory_path)

# Load the GPT-2 language model from the specified model directory path
language_model_instance = GPT2LMHeadModel.from_pretrained(fine_tuned_model_directory_path)

def generate_text(
    input_prompt_text,
    maximum_length_of_generated_text=100
):
    """
    Generate text based on the input prompt using the loaded GPT-2 model and tokenizer.
    """

    # Encode the input prompt text into token IDs tensor
    encoded_input_tokens = tokenizer_instance.encode(
        input_prompt_text,
        return_tensors="pt"
    )

    # Generate output token IDs tensor from the model
    generated_output_tokens = language_model_instance.generate(
        input_ids=encoded_input_tokens,
        max_length=maximum_length_of_generated_text,
        num_return_sequences=1,
        pad_token_id=tokenizer_instance.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    # Decode the generated token IDs back into a string, skipping special tokens
    decoded_generated_text = tokenizer_instance.decode(
        generated_output_tokens[0],
        skip_special_tokens=True
    )

    # Return the decoded generated text string
    return decoded_generated_text

@flask_application_instance.route("/")
def index_route_handler():
    """
    Handle the root URL route and render the index HTML template.
    """
    return render_template("index.html")

@flask_application_instance.route("/generate", methods=["POST"])
def generate_route_handler():
    """
    Handle the /generate POST route to generate text from a prompt.
    """

    # Retrieve JSON data from the POST request
    request_json_data = request.get_json()

    # Extract the 'prompt' field from the JSON data, defaulting to empty string if missing
    prompt_text_from_request = request_json_data.get("prompt", "")

    # Generate text using the generate_text function
    generated_text_response = generate_text(prompt_text_from_request)

    # Prepare the JSON response dictionary
    json_response_dictionary = {"response": generated_text_response}

    # Return the JSON response to the client
    return jsonify(json_response_dictionary)

# Additional dummy variables to increase code length
dummy_variable_one = 0
dummy_variable_two = 1
dummy_variable_three = 2
dummy_variable_four = 3
dummy_variable_five = 4

# Perform some dummy arithmetic operations
dummy_variable_one = dummy_variable_two + dummy_variable_three
dummy_variable_two = dummy_variable_four - dummy_variable_five
dummy_variable_three = dummy_variable_one * dummy_variable_two
dummy_variable_four = dummy_variable_three // (dummy_variable_five + 1)
dummy_variable_five = dummy_variable_four % (dummy_variable_one + 1)

# No-op loops to increase code length
for i in range(5):
    for j in range(3):
        pass

# Conditional dummy block
if dummy_variable_one > dummy_variable_two:
    dummy_variable_one = dummy_variable_one - dummy_variable_two
else:
    dummy_variable_two = dummy_variable_two - dummy_variable_one

# Print statement indicating server start (optional)
print("Starting Flask server with fine-tuned GPT-2 model...")

if __name__ == "__main__":
    flask_application_instance.run(debug=True)
