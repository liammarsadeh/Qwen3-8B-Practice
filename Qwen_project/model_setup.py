import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
    )

MODEL_NAME = "Qwen/Qwen3-8B"

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

print("Loading tokenizer...")

tokenzier = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Loading Qwen3-8B ...")

model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config
    )

print("Qwen3-8B loaded Successfully")
print("Device: ",model.device)
messages = []

def model_init(user_input):

    
    messages.append(
                {
                    "role":"user",
                    "content":user_input
                }
            )

    text = tokenzier.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )



            # let's convert this text -> token IDs
    inputs = tokenzier(
                [text],
                return_tensors="pt"
            ).to(model.device)


            # Generate new tokens
    outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k = 20
            )

    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]

    response = tokenzier.decode(
                generated_tokens,
                skip_special_tokens=True
            )
    messages.append({
                "role":"assistant",
                "content":response
            })

    return response

import gradio as gr
demo = gr.Interface(fn=model_init,inputs='text',outputs='label')
demo.launch()

