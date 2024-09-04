# t5_model.py
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 사전 학습된 T5 모델과 토크나이저 로드
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

def summarize_text(text: str):
    # input_ids = tokenizer.encode(f'summarize: {text}', return_tensors='pt')
    # summary_ids = model.generate(input_ids, max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return '입력값: '+str