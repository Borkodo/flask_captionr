from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_text(prompt, model_path="fine_tuned_model"):
    # Initialize the GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Encode the prompt to tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(
        input_ids,
        max_length=100,  # You can adjust this
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        no_repeat_ngram_size=1,
        do_sample=True
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return generated_text


if __name__ == "__main__":
    while True:
        prompt = input("Please enter your prompt: ")
        generated_text = generate_text(prompt)
        print("Generated text:", generated_text)
