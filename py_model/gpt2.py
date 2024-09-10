import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from loguru import logger
import time

model = GPT2LMHeadModel.from_pretrained('models/gpt2_final_model/model/')
tokenizer = GPT2Tokenizer.from_pretrained('models/gpt2_final_model/model/tokenizer/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def gentext(input_text, max_new_tokens_=125):
    prompt = f"입력값: {input_text}\n출력값:"
    # 입력 텍스트를 토큰화
    text_input = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)

    # 텍스트 인자
    with torch.no_grad():
        outputs = model.generate(
            text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            max_new_tokens=max_new_tokens_,
            num_beams=1,
            no_repeat_ngram_size=30,
            top_k=50,
            top_p=0.95,
            temperature=1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id)
        
    # 예측 결과 디코딩
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(predicted_text)
    
    return predicted_text

# 테스트
if __name__ == "__main__":
    time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logger.add(f"result/gpt2/{time_}.log",format="{message}", level="INFO")
    print(gentext('공포, 모자, 빵', 125))