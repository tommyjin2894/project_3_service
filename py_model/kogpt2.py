import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import time
from loguru import logger

model = GPT2LMHeadModel.from_pretrained('models/kogpt2_final_model/model')
tokenizer = PreTrainedTokenizerFast.from_pretrained("models/kogpt2_final_model/model/tokenizer/",
                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                    pad_token='<pad>', mask_token='<mask>')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def gentext(input_text, max_new_tokens_=125):
    prompt = f"입력값: {input_text}\n출력값:"
    
    # 입력 텍스트를 토큰화
    text_input = tokenizer(prompt, return_tensors='pt', padding=True).to(device)

    # 모델 타입에 따라 다른 인자를 설정
    with torch.no_grad():
        # GPT-2의 텍스트 생성 설정
        outputs = model.generate(
            input_ids=text_input['input_ids'],
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

    logger.add(f"result/kogpt2/{time_}.log",format="{message}", level="INFO")
    gentext('공포, 모자, 빵', 125)