from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# Load the trained model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('models/model1')
tokenizer = AutoTokenizer.from_pretrained('models/model1')

def translate(model, inference_request, tokenizer=tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)
    return translation

inference_request = 'You idiot'
translated_text = translate(model, inference_request, tokenizer)

print("Translated Text:", translated_text)
