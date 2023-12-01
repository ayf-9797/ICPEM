# -*- coding: utf-8 -*-
# 引入相应的包 Importing libraries
import gc
import os
import pickle
import random

import transformers
from constra_loss import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
# Importing the T5 modules from huggingface/transformers
from transformers import TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer
from torch.utils.data import WeightedRandomSampler
from rich.table import Column, Table
from rich import box
from rich.console import Console
from tqdm import tqdm
import time
import jsonlines
# from pandas.io.json import json_normalize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.nn import DataParallel
# Setting up the device for GPU usage
from torch import cuda
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply
from sklearn.model_selection import KFold

device = 'cuda' if cuda.is_available() else 'cpu'

smooth = SmoothingFunction().method1
# define a rich console logger
console = Console(record=True)


class Prompt:
    def __init__(self, template_fileA, template_fileB):
        self.templateA = []
        self.templateB = []
        with open(template_fileA, 'r') as f_in:
            for line in f_in:
                temp = []
                line = line.rstrip()
                words = line.split('\t')
                for word in words:
                    temp.append(word)
                self.templateA.append(temp)
        with open(template_fileB, 'r') as f_in:
            for line in f_in:
                temp = []
                line = line.rstrip()
                words = line.split('\t')
                for word in words:
                    temp.append(word)
                self.templateB.append(temp)
        self.rand_list = list(range(len(self.templateA)))
        np.random.shuffle(self.rand_list)

    def __call__(self, m_title, s_title, *args):
        """
            根据已有特征返回一组prompt
            "template = [span0,conj,span1]"
            :param m_title: 主标题，一般指
            :param s_title:
            :param args:
            :return:
            """
        rand = self.rand_list[0]
        if s_title == '介绍' or s_title == '简介' or s_title == '定义' or s_title == '概念':
            span = self.templateA[rand]
            prompt = span[0] + m_title + span[1]
        else:
            span = self.templateB[rand]
            if len(m_title) < args[0]:
                prompt = span[0] + m_title + '的' + s_title + span[2]
            else:
                prompt = span[0] + m_title + '和' + s_title + span[2]
        self.gen_randint(rand)
        return prompt

    def __len__(self):
        return len(self.templateA)

    def gen_randint(self, discard):
        self.rand_list.remove(discard)
        np.random.shuffle(self.rand_list)

    def restart(self):
        self.rand_list = list(range(len(self.templateA)))


# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    # console.print(table) # TODO TODO TODO 


# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):
    """
    多卡负载均衡
    """

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        # print('len(inputs): ', str(len(inputs)))
        # print('self.device_ids[:len(inputs)]', str(self.device_ids[:len(inputs)]))

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # replicas = self.replicate(self.module, device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        # print('replicas:', str(len(replicas)))

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        if len(inputs) > 0:
            bsz = inputs[0].size(self.dim)
        elif kwargs:
            bsz = list(kwargs.values())[0].size(self.dim)
        else:
            raise ValueError("You must pass inputs to the model!")
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        # print('bsz: ', bsz)
        # print('num_dev: ', num_dev)
        # print('gpu0_bsz: ', gpu0_bsz)
        # print('bsz_unit: ', bsz_unit)
        # print('chunk_sizes: ', chunk_sizes)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


class DataSetClass(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, dataframe, tokenizer, source_len, target_len, source_text, target_text, pre_train=False,
            num_broken=0.15
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
        self.pre_train = pre_train
        self.num_broken = num_broken

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
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
        if self.pre_train:
            target_broken = target["input_ids"].clone().squeeze()
            target_len = len(target_text)
            if target_len >= 256:
                target_len = 255
            num_broken = int(self.num_broken * target_len)

            broken_mask_loc = list(range(target_len))
            random.shuffle(broken_mask_loc)
            broken_mask_loc = broken_mask_loc[:num_broken]
            target_broken[broken_mask_loc] = 0
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids_y = target["input_ids"].squeeze()
        # target_mask = target["attention_mask"].squeeze()

        # temp = {
        #     "source_ids": source_ids.to(device, dtype = torch.long),
        #     "source_mask": source_mask.to(device, dtype = torch.long),
        #     "target_ids": target_ids.to(device, dtype = torch.long),
        #     "target_ids_y": target_ids.to(device, dtype = torch.long),
        # }
        if self.pre_train:
            return {
                "source_ids": source_ids.to(device, dtype=torch.long),
                "source_mask": source_mask.to(device, dtype=torch.long),
                "target_ids": target_broken.to(device, dtype=torch.long),
                "target_ids_y": target_ids_y.to(device, dtype=torch.long),
            }
        else:
            return {
                "source_ids": source_ids.to(device, dtype=torch.long),
                "source_mask": source_mask.to(device, dtype=torch.long),
                "target_ids": target_ids_y.to(device, dtype=torch.long),
                "target_ids_y": target_ids_y.to(device, dtype=torch.long),
            }


class Contrast_dataset(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, dataframe, tokenizer, source_len, target_len, index_name, prompt_generator, num_neg, main_len
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
        self.num_neg = num_neg
        self.index_name = index_name
        self.p_gen = prompt_generator
        self.main_len = main_len
        self.real_len = len(dataframe)

    def __len__(self):
        """returns the length of dataframe"""

        return self.main_len

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        index = index % self.real_len
        neg_list = list(range(len(self.data)))
        neg_list.remove(index)
        neg_list = random.sample(neg_list, self.num_neg)
        key_a, key_b = self.data[self.index_name][index]
        ori_input = self.p_gen(key_a, key_b, 10)
        pos_input = self.p_gen(key_a, key_b, 10)
        self.p_gen.restart()
        neg_input = [self.p_gen(self.data[self.index_name][i][0], self.data[self.index_name][index][1], 10) for i in
                     neg_list]
        contrast = [pos_input] + neg_input

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(ori_input.split())
        target_text = [" ".join(sent.split()) for sent in contrast]

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            target_text,
            max_length=self.summ_len,
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        # source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        # target_mask = target["attention_mask"].squeeze()
        # temp = {
        #     "source_ids": source_ids.to(device, dtype = torch.long),
        #     "source_mask": source_mask.to(device, dtype = torch.long),
        #     "target_ids": target_ids.to(device, dtype = torch.long),
        #     "target_ids_y": target_ids.to(device, dtype = torch.long),
        # }
        return {
            "source_ids": source_ids.to(device, dtype=torch.long),
            # "source_mask": source_mask.to(device, dtype=torch.long),
            "target_ids": target_ids.to(device, dtype=torch.long),
            # "target_ids_y": target_ids.to(device, dtype=torch.long),
        }


def train(epoch, tokenizer, model, device, loader, con_loader, optimizer, model_params, pre_train=False):
    """
    训练模型

    """
    # model.gradient_checkpointing_enable()
    # real_batch_size // grad_accumulation

    # model = DataParallel(model)
    if con_loader is not None:
        count_c_loss = contrast_loss(num_neg=model_params['num_neg'])
    model = BalancedDataParallel(12 // 1, model, dim=0).cuda()
    model.cuda()
    model.train()
    time1 = time.time()
    k = 0
    # for data, contrast_data in zip(loader, con_loader):
    for data in loader:
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        y_label = data['target_ids_y']
        lm_labels = y_label[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0].mean()
        # if con_loader is not None:
        #     # contrast learning
        #     ori_sample = contrast_data['source_ids'].to(device, dtype=torch.long)
        #     ori_sample = ori_sample.repeat(4, 1)
        #     contrast_sample = contrast_data['target_ids'].to(device, dtype=torch.long).detach()
        #     contrast_sample = contrast_sample.reshape(ori_sample.shape[0], ori_sample.shape[1])
        #
        #     if isinstance(model, BalancedDataParallel):
        #         ori_sample = model.module.encoder(input_ids=ori_sample).last_hidden_state
        #     else:
        #         ori_sample = model.encoder(input_ids=ori_sample).last_hidden_state
        #
        #     with torch.no_grad():
        #         if isinstance(model, BalancedDataParallel):
        #             contrast_sample = model.module.encoder(input_ids=contrast_sample).last_hidden_state
        #         else:
        #             contrast_sample = model.encoder(input_ids=contrast_sample).last_hidden_state
        #
        #     c_loss = count_c_loss(ori_sample, contrast_sample)
        #     loss = outputs[0].mean() + 0.01 * c_loss.mean()
        # else:
        #     loss = outputs[0].mean()
        # 每100步打印日志
        if k % 100 == 0 and k != 0:
            time2 = time.time()
            print(k, "epoch:" + str(epoch) + "-loss:" + str(float(loss)) + ";each step's time spent:" + str(
                float(time2 - time1) / float(k + 0.0001)))
        k = k + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(tokenizer, model, device, loader):
    """用BLEU4评估"""
    model.eval()
    bleus = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0), desc='Evaluate'):
            target_ids = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=model_params["MAX_SOURCE_TEXT_LENGTH"],
                do_sample=True,
                top_p=0.6,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                      target_ids]
            bleu_score = sentence_bleu([list(p) for p in preds], list(target[0]), [0.25, 0.25, 0.25, 0.25],
                                       smoothing_function=smooth)
            bleus.append(bleu_score)
        return sum(bleus) / len(bleus)


# t5模型训练
def T5Trainer(
        dataframe, source_text, target_text, model_params, output_dir="./outputs/", contrast_data=None
):
    """
    T5 trainer
    """
    pre_train = model_params['Pre_train']
    k_fold = model_params['K_fold']
    # PART_TRAIN = True
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    console.log(f"[Data]: Reading data...\n")
    dataframe = dataframe[[source_text, target_text]]  # , 'type'
    # if PART_TRAIN:
    #     part_num = 10
    #     for i in range(part_num):
    #         pass
    #     part_rate = 0.1
    #     train_all = dataframe.sample(frac=part_rate, random_state=model_params["SEED"])
    #     train_all = pd.concat([train_all, temp_df], ignore_index=True)
    #     train_dataset = train_all.sample(frac=0.94, random_state=model_params["SEED"])
    #     val_dataset = train_all.drop(train_dataset.index).reset_index(drop=True)
    #     training_set = DataSetClass(
    #         train_dataset,
    #         tokenizer,
    #         model_params["MAX_SOURCE_TEXT_LENGTH"],
    #         model_params["MAX_TARGET_TEXT_LENGTH"],
    #         source_text,
    #         target_text,
    #     )
    #     val_set = DataSetClass(
    #         val_dataset,
    #         tokenizer,
    #         model_params["MAX_SOURCE_TEXT_LENGTH"],
    #         model_params["MAX_TARGET_TEXT_LENGTH"],
    #         source_text,
    #         target_text,
    #     )
    #     train_params = {
    #         # "sampler":train_sampler,
    #         "batch_size": model_params["TRAIN_BATCH_SIZE"],
    #         "shuffle": True,
    #         "num_workers": 0,
    #     }
    #
    #     val_params = {
    #         "batch_size": model_params["VALID_BATCH_SIZE"],
    #         "shuffle": False,
    #         "num_workers": 0,
    #     }
    #
    #     training_loader = DataLoader(training_set, **train_params)
    #     val_loader = DataLoader(val_set, **val_params)
    #
    #     optimizer = torch.optim.Adam(
    #         params=model.parameters(), lr=model_params["LEARNING_RATE"]
    #     )
    #
    #     console.log(f"[Initiating Fine Tuning]...\n")
    #     best_bleu = 0
    #     for epoch in range(model_params["TRAIN_EPOCHS"]):
    #         train(epoch, tokenizer, model, device, training_loader, optimizer)
    #
    #         path = os.path.join(output_dir, "model_files")
    #         tokenizer.save_pretrained(path)
    #         console.log(f"[Initiating Validation]...\n")
    #         with torch.no_grad():
    #             cur_bleu = evaluate(tokenizer, model, device, val_loader)
    #             if cur_bleu > best_bleu:
    #                 console.log(f"[Saving Model]...\n")
    #                 model.save_pretrained(path)
    #                 best_bleu = cur_bleu
    #             else:
    #                 os.mkdir(str(epoch))
    #                 low_path = os.path.join(str(epoch), 'model_files')
    #                 model.save_pretrained(low_path)
    #             print('Best bleu: {}, Current bleu: {}'.format(best_bleu, cur_bleu))
    temp_fold_train = []
    temp_fold_val = []

    if k_fold is None:
        train_size = 0.8
        train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
        val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)
        temp_fold_train.append(train_dataset)
        temp_fold_val.append(val_dataset)
    else:
        kf = KFold(n_splits=k_fold, random_state=model_params["SEED"], shuffle=True)
        for train_index, val_index in kf.split(dataframe):
            train_dataset = dataframe.loc[train_index].reset_index(drop=True)
            val_dataset = dataframe.loc[val_index].reset_index(drop=True)
            temp_fold_train.append(train_dataset)
            temp_fold_val.append(val_dataset)
    cur_fold = 0
    for train_dataset, val_dataset in zip(temp_fold_train, temp_fold_val):
        cur_fold = cur_fold + 1
        # logging
        console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")
        tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])  # rT5Tokenizer
        model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
        model = model.to(device)

        # 打印数据集相关日志：数据量、训练步数
        console.print(f"FULL Dataset: {dataframe.shape}")
        console.print(f"TRAIN Dataset: {train_dataset.shape}")
        console.print(f"TEST Dataset: {val_dataset.shape}\n")
        total_train_steps = int(
            (train_dataset.shape[0] * model_params["TRAIN_EPOCHS"]) / model_params["TRAIN_BATCH_SIZE"])
        console.print(f"Total Train Steps: {total_train_steps}\n")
        if contrast_data is not None:
            prompt_generator = Prompt('../templatea.txt', '../templateb.txt')
            contrast_dataset = Contrast_dataset(contrast_data,
                                                tokenizer,
                                                model_params["MAX_SOURCE_TEXT_LENGTH"],
                                                model_params["MAX_TARGET_TEXT_LENGTH"],
                                                'input',
                                                prompt_generator,
                                                model_params['num_neg'],
                                                len(train_dataset)
                                                )
            contrast_loader = DataLoader(contrast_dataset, batch_size=8)
        training_set = DataSetClass(
            train_dataset,
            tokenizer,
            model_params["MAX_SOURCE_TEXT_LENGTH"],
            model_params["MAX_TARGET_TEXT_LENGTH"],
            source_text,
            target_text,
            pre_train=pre_train
        )
        val_set = DataSetClass(
            val_dataset,
            tokenizer,
            model_params["MAX_SOURCE_TEXT_LENGTH"],
            model_params["MAX_TARGET_TEXT_LENGTH"],
            source_text,
            target_text,
            pre_train=False
        )
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
        val_loader = DataLoader(val_set, **val_params)

        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=model_params["LEARNING_RATE"]
        )

        console.log(f"[Initiating Fine Tuning]...\n")
        best_bleu = 0
        for epoch in range(model_params["TRAIN_EPOCHS"]):

            train(epoch, tokenizer, model, device, training_loader, None, optimizer, model_params, False)

            path = os.path.join(output_dir, "model_files")
            path = path + 'fold%d' % cur_fold
            console.log(f"[Initiating Validation]...\n")
            with torch.no_grad():
                cur_bleu = evaluate(tokenizer, model, device, val_loader)
                print(cur_bleu)
                if cur_bleu > best_bleu:
                    console.log(f"[Saving Model]...\n")
                    model.save_pretrained(path)
                    tokenizer.save_pretrained(path)
                    best_bleu = cur_bleu
                else:
                    if os.path.exists(str(epoch)):
                        console.print('overwrite for %d' % epoch)
                    else:
                        os.mkdir(str(epoch))
                    low_path = os.path.join(str(epoch), 'model_files')
                    low_path =  low_path + 'fold%d' % cur_fold
                    model.save_pretrained(low_path)
                    tokenizer.save_pretrained(low_path)
                print('Best bleu: {}, Current bleu: {}'.format(best_bleu, cur_bleu))

        console.save_text(os.path.join(output_dir, "logs.txt"))

        console.log(f"[Validation Completed.]\n")
        console.print(
            f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
        )
        console.print(
            f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions.csv')}\n"""
        )
        console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")
        if 'train_dataset' in locals():
            del train_dataset
        if 'val_dataset' in locals():
            del val_dataset
        gc.collect()
        print("%d fold complete and memory clear" % cur_fold)


def load_mydata(folder):
    file_list = os.listdir(folder)
    prompt = []
    sent = []
    for file in file_list:
        with open(folder + '/' + file, 'rb') as f_in:
            temp = pickle.load(f_in)
            prompt.append((temp['mtitle'], temp['stitle']))
            sent.append(temp['sl'])
    return prompt, sent


if "__main__" == __name__:

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    print(torch.__version__)
    print(torch.cuda.is_available())
    # 定义模型的参数
    # ICPEM
    # 初始模型路径,pre_train:"../Chinese_Chat_T5_Base", domain
    # adapted:"Pre_trained_model/model_files"
    # Pclue
    # '../PromptCLUE-base-v1-5'
    # GPT

    model_params = {
        "MODEL": "../PromptCLUE-base-v1-5",  #
        "Pre_train": False,
        "K_fold": 5,
        "TRAIN_BATCH_SIZE": 32,  # 训练batch size
        "VALID_BATCH_SIZE": 64,  # 评估batch size
        "TRAIN_EPOCHS": 5,  # 训练epoch数
        "LEARNING_RATE": 1e-4,  # 学习率
        "MAX_SOURCE_TEXT_LENGTH": 256,  # 句子最大长度
        "MAX_TARGET_TEXT_LENGTH": 256,  # 标签最大长度
        'num_neg': 3,
        "SEED": 100,  # 随机种子
    }
    # tdf = pd.DataFrame()
    # tdf['input'] = [{'a': 5}, {'c': 6}]
    # tdf['target'] = [{'a': 5}, {'c': 6}]
    # qdf = pd.DataFrame()
    # qdf['input'] = [{'a': 5}, {'c': 6}]
    # # tdf = pd.concat([tdf, qdf],ignore_index=True)
    # print(tdf[0])
    #
    # qdf = tdf[['input','target']]
    # print(qdf[0])

    input = []
    target = []

    if model_params['Pre_train']:
        with jsonlines.open('../BELLE/train/data_dir/Belle_open_source_1M.train_ex.json', 'r') as f:
            for l in f:
                input.append(l['instruction'])
                target.append(l['output'])
    else:
        temp_ = pd.read_csv('../cancer_dialog.csv', encoding='gb18030')
        for i in range(len(temp_)):
            input.append(temp_['title'][i])
            target.append(temp_['answer'][i])
    df = pd.DataFrame()

    df['input'] = input
    df['target'] = target
    print(df.shape)

    # p, s = load_mydata('../my_data')
    # df_contrast = pd.DataFrame()
    # df_contrast['input'] = p
    # df_contrast['target'] = s
    T5Trainer(
        dataframe=df,
        source_text="input",
        target_text="target",
        model_params=model_params,
        output_dir="./domain/",
        contrast_data=None,
    )
