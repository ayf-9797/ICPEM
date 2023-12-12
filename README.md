# ICPEM
The Pre-training and Domain adaption of ICPEM  
  
To get started, the model initialization parameters from Hugging Face and github need to get.  
Bloom: https://www.huggingface.co/yuanzhoulvpi/chinese_bloom_7b_chat_v2  
GPT: https://github.com/fastnlp/CPT  
Prompt-clue: https://huggingface.co/ClueAI/PromptCLUE-base-v1-5  
ICPEM:https://huggingface.co/mxmax/Chinese_Chat_T5_Base  
  
The dataset also needs to be obtained from the internet where   
BELLE: https://github.com/LianjiaTech/BELLE  
doctor-patient dialogue: https://github.com/Toyhom/Chinese-medical-dialogue-data  
Then, Bloom and GPT can be trained directly by the given code, and the code of T5 can accomplish pre-training of ICPEM, domain adaptation, and domain adaptation of Prompt-clue by only modifying the model path.  

The operating environment needs to conform:  
numpy                         1.26.0  
torch                         2.0.1  
transformers                  4.34.1  
tokenizers                    0.14.1  
cuda                          11.6  

The decoding parameters used in this paper are as follows:
ICPEM, Prompt_clue:  
generated_ids = model.generate(  
                input_ids=ids,  
                attention_mask=mask,  
                max_length=256,  
                do_sample=True,  
                top_p=0.6,  
                early_stopping=True  
            )  
GPT, Bloom:  
generated_ids = model.generate(  
                input_ids=data['input_ids'],  
                attention_mask=data['attention_mask'],  
                min_length=3,  
                max_new_tokens=256,  
                pad_token_id=tokenizer.pad_token_id,  
                repetition_penalty=3.5,  
                length_penalty=2.5,  
                early_stopping=True  
            )
