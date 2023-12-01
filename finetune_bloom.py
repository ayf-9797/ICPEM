import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import transformers
import torch
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rich.console import Console
from transformers import AutoConfig
from peft import PromptTuningConfig, TaskType, PromptTuningInit, get_peft_model
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq
from transformers import BloomTokenizerFast
from torch.utils.data import Dataset, DataLoader
import jsonlines
from train import make_train_dataset
import pandas as pd
import numpy as np


smooth = SmoothingFunction().method1
# define a rich console logger
console = Console(record=True)

class DataSetClass(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, dataframe, tokenizer, source_len, target_len, source_text, target_text,trainable=True
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
def train(epoch, tokenizer, model, device, loader, optimizer, ckpt=False, grad_accumulation = 4):
    """
    训练模型

    """
    if ckpt:
        model.gradient_checkpointing_enable()
    # model = BalancedDataParallel(14 // 2, model, dim=0).cuda()
    # model.cuda()
    model.train()
    time1 = time.time()
    optimizer.zero_grad()
    for step, data in enumerate(loader, 0):

        # y = data["source_ids"].to(device, dtype=torch.long)
        # mask = data["source_mask"].to(device, dtype=torch.long)
        # type_token = data['type_token'].to(device, dtype=torch.long)
        data = {k: v.to(device) for k, v in data.items()}
        outputs = model(**data)
        loss = outputs[0].mean()
        # 每100步打印日志
        if step % 100 == 0 and step != 0:
            time2 = time.time()
            print(step, "epoch:" + str(epoch) + "-loss:" + str(float(loss)) + ";each step's time spent:" + str(
                float(time2 - time1) / float(step + 0.0001)))
        loss /= grad_accumulation
        loss.backward()
        if (step+1) % grad_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()


def evaluate(tokenizer, model, device, loader):
    """用BLEU4评估"""
    tokenizer.padding_side = 'left'
    model.eval()
    bleus = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0), desc='Evaluate'):
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
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                      data['labels']]
            bleu_score = sentence_bleu([list(p) for p in preds], list(target[0]), [0.25, 0.25, 0.25, 0.25],
                                       smoothing_function=smooth)
            bleus.append(bleu_score)
    tokenizer.padding_side = 'right'
    return sum(bleus) / len(bleus)

def BLOOM_trainer(
        dataframe, source_text, target_text, model_params, output_dir="./outputs/", peft_id='4/model_files'):
    # torch.cuda.set_device(0)
    # PART_TRAIN = True
    device = torch.device('cuda:0')
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # model_name_or_path = 'chinese_bloom_7b_chat_v2'
    # model_max_length = 512
    # optim = "adamw_torch"
    # peft_id = None
    # data_path = 'data_folder'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_params['MODEL'],
        cache_dir=None,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=False
    )
    if peft_id is None:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=42,
            prompt_tuning_init_text="我想请你扮演医生的角色，为我回答以下问题：\n",
            tokenizer_name_or_path=model_params['MODEL'],
        )
        model = get_peft_model(model, peft_config)
    # training_args = AutoConfig.from_pretrained(model_name_or_path)
    # model.is_parallelizable = True
    # model.model_parallel = True
    torch.cuda.empty_cache()
    tokenizer = transformers.BloomTokenizerFast.from_pretrained(
        model_params['MODEL'],
        cache_dir=None,
        model_max_length=model_params['MAX_SOURCE_TEXT_LENGTH'],
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.print_trainable_parameters()
    # model = DataParallel(model)
    model.to(device)
    console.log(f"[Data]: Reading data...\n")
    dataframe = dataframe[[source_text, target_text]]  # , 'type'
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # 打印数据集相关日志：数据量、训练步数
    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")
    total_train_steps = int((train_dataset.shape[0] * model_params["TRAIN_EPOCHS"]) / model_params["TRAIN_BATCH_SIZE"])
    console.print(f"Total Train Steps: {total_train_steps}\n")

    training_set = DataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
        trainable=True
    )
    # val_set = DataSetClass(
    #     val_dataset,
    #     tokenizer,
    #     model_params["MAX_SOURCE_TEXT_LENGTH"],
    #     model_params["MAX_TARGET_TEXT_LENGTH"],
    #     source_text,
    #     target_text,
    #     trainable=False
    # )
    train_params = {
        # "sampler":train_sampler,
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    training_loader = DataLoader(training_set, **train_params)
    #val_loader = DataLoader(val_set, **val_params)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    console.log(f"[Initiating Fine Tuning]...\n")
    best_bleu = 0
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        path = os.path.join(output_dir, "model_files")
        tokenizer.save_pretrained(path)
        console.log(f"[Initiating Validation]...\n")
        # with torch.no_grad():
        #     cur_bleu = evaluate(tokenizer, model, device, val_loader)
        #     if cur_bleu > best_bleu:
        #         console.log(f"[Saving Model]...\n")
        #         model.save_pretrained(path)
        #         best_bleu = cur_bleu
        #     else:
        #         if os.path.exists(str(epoch)):
        #             console.print('overwrite for %d' % epoch)
        #         else:
        #             os.mkdir(str(epoch))
        #         low_path = os.path.join(str(epoch), 'model_files')
        #         model.save_pretrained(low_path)
        #     print('Best bleu: {}, Current bleu: {}'.format(best_bleu, cur_bleu))
        if os.path.exists(str(epoch)):
            console.print('overwrite for %d' % epoch)
        else:
            os.mkdir(str(epoch))
        low_path = os.path.join(str(epoch), 'model_files')
        model.save_pretrained(low_path)
    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    # 定义模型的参数
    model_params = {
        "MODEL": "chinese_bloom_7b_chat_v2",  # 初始模型路径
        "TRAIN_BATCH_SIZE": 4,  # 训练batch size
        "VALID_BATCH_SIZE": 4,  # 评估batch size
        "TRAIN_EPOCHS": 7,  # 训练epoch数
        "LEARNING_RATE": 1e-4,  # 学习率
        "MAX_SOURCE_TEXT_LENGTH": 250,  # 句子最大长度
        "MAX_TARGET_TEXT_LENGTH": 250,  # 标签最大长度
        "SEED": 100,  # 随机种子
    }

    input_ = []
    target_ = []
    with jsonlines.open('data_folder/cancer_dialog.json') as f_in:
        for line in f_in:
            input_.append(line['input'])
            target_.append(line['target'])
    df = pd.DataFrame()
    df['input'] = input_
    df['target'] = target_
    print(df.shape)
    BLOOM_trainer(
        dataframe=df,
        source_text="input",
        target_text="target",
        model_params=model_params,
        output_dir="chinese_bloom_7b_chat_v2/outputs",
        peft_id=None
    )
