import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import time
import csv
import random
import string

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper

@measure_time
def generate_tokens(model, tokenizer, input_ids, max_length, num_return_sequences, attention_mask):
    return model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

def random_string(length):
    return ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=length))

def main(configs, output_file):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16")
    model.to_bettertransformer()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is empty
        file.seek(0, 2)  # Move to the end of the file
        if file.tell() == 0:
            writer.writerow(["batch_size", "input_token_length", "output_token_length", "prefill_time", "decode_time"])

        for config in configs:
            batch_size = config['batch_size']
            input_token_length = config['input_token_length']
            output_token_length = config['output_token_length']

            dummy_input = " ".join([random_string(5) for _ in range(input_token_length)])
            input_ids = tokenizer(dummy_input, return_tensors="pt", truncation=True, max_length=input_token_length).input_ids.to(device)
            input_ids = input_ids[:, :input_token_length]  # Truncate if necessary
            input_ids = input_ids.repeat(batch_size, 1)

            attention_mask = torch.ones_like(input_ids).to(device)

            _, prefill_time = measure_time(model)(input_ids)
            _, decode_time = generate_tokens(model, tokenizer, input_ids, output_token_length, batch_size, attention_mask)

            writer.writerow([batch_size, input_token_length, output_token_length, prefill_time, decode_time])

            print(f"Batch size {batch_size}, Input tokens {input_token_length}, Output tokens {output_token_length}: Prefill time = {prefill_time:.4f} seconds, Decode time = {decode_time:.4f} seconds")

if __name__ == "__main__":
    configs = [
        # {"batch_size": 1, "input_token_length": 32, "output_token_length": 16},
        # {"batch_size": 1, "input_token_length": 32, "output_token_length": 32},
        # {"batch_size": 1, "input_token_length": 32, "output_token_length": 64},
        # {"batch_size": 1, "input_token_length": 32, "output_token_length": 128},
        # {"batch_size": 1, "input_token_length": 32, "output_token_length": 256},
        # {"batch_size": 1, "input_token_length": 32, "output_token_length": 512},
        # {"batch_size": 1, "input_token_length": 32, "output_token_length": 1024},
        # {"batch_size": 1, "input_token_length": 16, "output_token_length": 32},
        # {"batch_size": 1, "input_token_length": 32, "output_token_length": 32},
        # {"batch_size": 1, "input_token_length": 64, "output_token_length": 32},
        # {"batch_size": 1, "input_token_length": 128, "output_token_length": 32},
        {"batch_size": 1, "input_token_length": 256, "output_token_length": 32},
        {"batch_size": 1, "input_token_length": 512, "output_token_length": 32},
        {"batch_size": 1, "input_token_length": 1024, "output_token_length": 32},
    ]
    output_file = "timings.csv"

    main(configs, output_file)
