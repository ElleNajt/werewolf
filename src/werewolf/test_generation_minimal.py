"""Minimal test for multi-GPU generation - no detector, just model loading and generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "deception-detection"))

import torch
from deception_detection.models import ModelName, get_llama3_model_and_tokenizer

print("=" * 80)
print("Minimal Generation Test")
print("=" * 80)

print("\n1. Loading model...")
model, tokenizer = get_llama3_model_and_tokenizer(
    ModelName.LLAMA_70B_3_3, omit_model=False
)
print("✅ Model loaded!\n")

print("2. Checking device placement...")
device_params = {}
for name, param in model.named_parameters():
    device = str(param.device)
    if device not in device_params:
        device_params[device] = 0
    device_params[device] += param.numel()

for device, count in sorted(device_params.items()):
    params_b = count / 1e9
    print(f"   {device}: {params_b:.2f}B parameters")
print()

print("3. Testing generation...")
test_prompt = "You are playing Werewolf. Say hello in JSON format with 'message' field."

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": test_prompt}
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(f"   Prompt: {test_prompt[:60]}...")

# Tokenize and move to cuda
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
input_length = inputs['input_ids'].shape[1]

print(f"   Input tokens: {input_length}")
print(f"   Generating with greedy decoding...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,  # Greedy decoding
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(
    outputs[0][input_length:],
    skip_special_tokens=True
)

print(f"\n✅ Generation successful!")
print(f"Response: {response[:200]}...")
print("\n" + "=" * 80)
print("SUCCESS - Greedy decoding works!")
print("=" * 80)
