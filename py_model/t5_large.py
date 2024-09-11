import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import time
from loguru import logger



def gentext(input_text, max_new_tokens_=64):
    model = T5ForConditionalGeneration.from_pretrained('models/t5_large_large/model/')
    tokenizer = T5TokenizerFast.from_pretrained('models/t5_large_large/model/tokenizer')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    prompt = input_text
    # 입력 텍스트를 토큰화
    text_input = tokenizer(prompt, return_tensors='pt', padding=True).to(device)

    # 모델 타입에 따라 다른 인자를 설정
    with torch.no_grad():
        # GPT-2의 텍스트 생성 설정
        outputs = model.generate(
            text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            max_new_tokens=max_new_tokens_,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # 예측 결과 디코딩
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    logger.info(f'입력 : {input_text} \n출력: {predicted_text}')
    
    return predicted_text

# 테스트
if __name__ == "__main__":
    time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    logger.add(f"result/t5_large/{time_}.log",format="{message}", level="INFO")
    gentext('공포, 모자, 빵', 64)