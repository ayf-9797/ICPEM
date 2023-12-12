import gc
import multiprocessing
import time
import jsonlines
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
from tqdm import tqdm
import pandas as pd
# from fengshen_gpt2main.GPT_train import DataSetClass as GPT_dataset
# from Chinese_Chat_T5_Base_main.t5_train import DataSetClass as T5_dataset
import numpy as np
import jieba  # you can use any other word cutting library
import os
import importlib
# from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2LMHeadModel, \
#     GPT2Tokenizer, BloomForCausalLM, BloomTokenizerFast
import transformers
import peft
smooth = SmoothingFunction().method1

# from peft import PeftConfig, PeftModel


class T5_dataset(torch.utils.data.Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index, device='cuda'):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()
        # temp = {
        #     "source_ids": source_ids.to(device, dtype = torch.long),
        #     "source_mask": source_mask.to(device, dtype = torch.long),
        #     "target_ids": target_ids.to(device, dtype = torch.long),
        #     "target_ids_y": target_ids.to(device, dtype = torch.long),
        # }
        return {
            "source_ids": source_ids.to(device, dtype=torch.long),
            "source_mask": source_mask.to(device, dtype=torch.long),
            "target_ids": target_ids.to(device, dtype=torch.long),
            "target_ids_y": target_ids.to(device, dtype=torch.long),
        }


class GPT_dataset(torch.utils.data.Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, dataframe, tokenizer, source_len, target_len, source_text, target_text, trainable=True
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.trainable = trainable
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index, device='cuda'):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())
        if self.trainable:
            input_text = source_text + target_text
            gpt_input = self.tokenizer.batch_encode_plus(
                [input_text],
                max_length=self.source_len,
                pad_to_max_length=True,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            input_ids = gpt_input["input_ids"].squeeze()
            input_mask = gpt_input["attention_mask"].squeeze()
            return {
                "input_ids": input_ids.to(device),
                "attention_mask": input_mask.to(device, dtype=torch.long),
                "labels": input_ids.to(device)
            }

        else:
            gpt_input = self.tokenizer.batch_encode_plus(
                [source_text],
                max_length=self.source_len,
                pad_to_max_length=True,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
            gpt_label = self.tokenizer.batch_encode_plus(
                [target_text],
                max_length=self.source_len,
                pad_to_max_length=True,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
            # gpt_type_token = [0] * self.source_len + [1] * self.summ_len
            input_ids = gpt_input["input_ids"].squeeze()
            input_mask = gpt_input["attention_mask"].squeeze()
            # "type_token": gpt_type_token.to(device, dtype=torch.long),
            # gpt_type_token = torch.LongTensor(gpt_type_token)
            label_ids = gpt_label["input_ids"].squeeze()
            return {
                "input_ids": input_ids.to(device),
                "attention_mask": input_mask.to(device, dtype=torch.long),
                "labels": label_ids.to(device)
            }


@torch.no_grad()
def generate_text(df, model_name, result_queue, mode):
    device_ids = [torch.device('cuda:0'), torch.device('cuda:1')]
    process = multiprocessing.current_process()
    process_id = process.name.split("-")[-1]
    process_id = int(process_id) - 1
    device = device_ids[process_id]
    pred_fp = open(model_name + 'preds%d.txt' % process_id, 'w+')
    targ_fp = open('traget%d.txt' % process_id, 'w+')
    if mode == 't5':
        # 初始化tokenizer和模型
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        val_dataset = T5_dataset(
            df,
            tokenizer,
            256,
            256,
            'input',
            'target',
        )
    elif mode == 'gpt':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name,
                                                               padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        model = peft.PeftModel.from_pretrained(model, '5/model_files')
        val_dataset = GPT_dataset(
            df,
            tokenizer,
            250,
            250,
            'input',
            'target',
            trainable=False
        )
    elif mode == 'bloom':
        tokenizer = transformers.BloomTokenizerFast.from_pretrained(model_name,
                                                                    padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = transformers.BloomForCausalLM.from_pretrained(model_name, device_map='auto',
                                                              torch_dtype=torch.bfloat16, load_in_8bit=True)
        model = peft.PeftModel.from_pretrained(model, '5/model_files')
        val_dataset = GPT_dataset(
            df,
            tokenizer,
            250,
            250,
            'input',
            'target',
            trainable=False
        )
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=24)
    tbar = tqdm(total=len(loader), desc='Evaluate')
    model.eval()
    model = model.to(device)
    print('*****************************************\nProcess id%d:running' % process_id)
    for _, data in enumerate(loader, 0):
        if mode == 't5':
            target_ids = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=256,
                do_sample=True,
                top_p=0.6,
                early_stopping=True
            )
        elif mode == 'gpt' or mode == 'bloom':
            target_ids = data['labels'].to(device, dtype=torch.long)
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                min_length=3,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=3.5,
                length_penalty=2.5,
                early_stopping=True
            )
        preds_ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                  generated_ids]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in target_ids]
        preds = [i if i != '' else '抱歉，我没听懂，请您换种说法重新提问。' for i in preds_]
        for i, j in zip(preds, target):
            pred_fp.write(i + '\n')
            targ_fp.write(j + '\n')
        tbar.update(1)


def json2dataframe(load_mode, source_file, model_type, select_top=-1):
    lines = jsonlines.open(source_file, 'r')
    if select_top != -1:  # select_top==-1 -->全量预测；其他值，则选取top值进行预测
        lines = lines[0:select_top]
        print("length of lines:", len(lines))
    input_ = []
    target = []
    type = []
    for i, line in enumerate(lines):
        # print(i,line)
        if load_mode == 'ct5':
            input_.append(line['instruction'])
            target.append(line['output'])
        elif load_mode == 'pclue':
            if model_type != 'ct5':
                input_.append(line["input"] + '答案：')
                target.append(line["target"])
            else:
                input_.append(line["input"])
                target.append(line["target"])
            # type.append(line["type"])

    df = pd.DataFrame()
    df['input'] = input_
    df['target'] = target
    return df


@torch.no_grad()
def gen_text_count(df, model_name, result_queue, mode, batch_size):
    device_ids = [torch.device('cuda:0'), torch.device('cuda:1')]
    process = multiprocessing.current_process()
    process_id = process.name.split("-")[-1]
    process_id = int(process_id) % 2 - 1
    device = device_ids[process_id]
    if mode == 't5':
        max_len = 256
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        val_dataset = T5_dataset(
            df,
            tokenizer,
            max_len,
            max_len,
            'input',
            'target',
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
    elif mode == 'gpt' or mode == 'bloom':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name,
                                                               padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        model = peft.PeftModel.from_pretrained(model, '5/model_files')
        max_len = 250
        val_dataset = GPT_dataset(
            df,
            tokenizer,
            max_len,
            max_len,
            'input',
            'target',
            trainable=False
        )
    bleusa = []
    bleusb = []
    bleusc = []
    bleusd = []
    rouge = Rouge()
    rouge_gradea = []
    rouge_gradeb = []
    rouge_gradel = []
    model.to(device)
    model.eval()
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    print(len(loader))
    for data in tqdm(loader, desc='Evaluate'):
        target_ids = data['target_ids'].to(device, dtype=torch.long)
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)
        generated_ids = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_length=max_len,
            do_sample=True,
            top_p=0.6,
            early_stopping=True
        )
        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                 generated_ids]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                  target_ids]
        preds = [i if i != '' else '抱歉，我没听懂，请您换种说法重新提问。' for i in preds]
        for i, j in zip(preds, target):
            try:
                pred_ = ' '.join(jieba.cut(i))
                target_ = ' '.join(jieba.cut(j))
                scores = rouge.get_scores(pred_, target_)
                score_ra = scores[0]["rouge-1"]["f"]
                score_rb = scores[0]["rouge-2"]["f"]
                score_rl = scores[0]["rouge-l"]["f"]
                rouge_gradea.append(score_ra)
                rouge_gradeb.append(score_rb)
                rouge_gradel.append(score_rl)
            except:
                print('error sentence is' + i)
                rouge_gradea.append(0.0)
                rouge_gradeb.append(0.0)
                rouge_gradel.append(0.0)
        bleu_scorea = sentence_bleu([list(p) for p in preds], list(target[0]), [1, 0, 0, 0],
                                    smoothing_function=smooth)
        bleu_scoreb = sentence_bleu([list(p) for p in preds], list(target[0]), [0, 1, 0, 0],
                                    smoothing_function=smooth)
        bleu_scorec = sentence_bleu([list(p) for p in preds], list(target[0]), [0, 0, 1, 0],
                                    smoothing_function=smooth)
        bleu_scored = sentence_bleu([list(p) for p in preds], list(target[0]), [0, 0, 0, 1],
                                    smoothing_function=smooth)
        bleusa.append(bleu_scorea)
        bleusb.append(bleu_scoreb)
        bleusc.append(bleu_scorec)
        bleusd.append(bleu_scored)

    result_queue.put((process_id, sum(bleusa) / len(bleusa), sum(bleusb) / len(bleusb), sum(bleusc) / len(bleusc), \
                      sum(bleusd) / len(bleusd), sum(rouge_gradea) / len(rouge_gradea),
                      sum(rouge_gradeb) / len(rouge_gradeb), \
                      sum(rouge_gradel) / len(rouge_gradel)))


if __name__ == '__main__':
    # model_type = 'bloom'
    multiprocessing.set_start_method('spawn')

    model_type = 't5'
    num_processes = 2
    # 设置生成文本的输入数据
    # df = json2dataframe('pclue', 'BELLE/train/data_dir/Belle_open_source_1M.dev.json', 'pclue')
    df = json2dataframe('pclue', 'data_folder/cancer_dialog.json', 'ct5')
    # df = df.sample(frac=0.001)
    dfa = df.sample(frac=0.5, random_state=100)
    dfb = df.drop(dfa.index).reset_index(drop=True)
    dfa = dfa.reset_index(drop=True)
    data_list = [dfa, dfb]
    fold_model = ['model_filesfold1', 'model_filesfold2', 'model_filesfold3', 'model_filesfold4','model_filesfold5']
    for model_name in fold_model:
        out_name = 'Chinese_Chat_T5_Base_main/Pclue/domain'
        model_name = os.path.join(out_name, model_name)
        result_queue = multiprocessing.Queue()
        temp_result = []
        # 并行运算
        processes = []
        # model_name = 'chinese_bloom_7b_chat_v2'
        for i in range(num_processes):
            temp = data_list[i]
            p = multiprocessing.Process(target=gen_text_count, args=(temp, model_name, result_queue, model_type, 64))
            processes.append(p)
            p.start()
            if i == 0:
                time.sleep(100)
                print('************************\ntime enough to get the next process')
            else:
                print('all process start')
        for p in processes:
            p.join()
            process_id, a, b, c, d, e, f, g = result_queue.get()
            temp_result.append(np.array([a, b, c, d, e, f, g]))
            print('over process-----****')
        a, b, c, d, e, f, g = (temp_result[0] + temp_result[1]) / 2
        with open('out_grade', 'a') as out_fp:
            out_fp.write(model_name + ':BLEU are%.3f,%.3f,%.3f,%.3f\nROUGE are %.3f,%.3f,%.3f' % (a, b, c, d, e, f, g))
        gc.collect()
