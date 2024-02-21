from transformer_mlm import Custom_model
from transformers import (
    set_seed,
)
from abc import ABC, abstractmethod
import torch
from datasets import Dataset
from typing import Optional, Any, Dict, List, NewType, Tuple
from dataclasses import dataclass, field
import numpy as np
from eval_func import compute_metrics_func
from tqdm import tqdm


class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    # def collate_batch(self) -> Dict[str, torch.Tensor]:
    def __call__(self) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.

        Returns:
            A dictionary of tensors
        """
        pass


@dataclass
class DataCollatorForLanguageModeling(DataCollator):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, entity_rel_dict, remove_head):
        self.entity_rel_dict = entity_rel_dict
        self.remove_head = remove_head

    # def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    def __call__(self, examples) -> Dict[str, torch.Tensor]:
		
        # now we have to pad all sentences to max length and also truncate longer ones
		# first truncate
        tokenized_lines = [example['input_ids'] for example in examples]

        tokenized_lines = np.array(tokenized_lines, dtype=int)

        # tensorizing
        tokenized_lines = [torch.tensor(line) for line in tokenized_lines]
        batch = torch.stack(tokenized_lines, dim=0)

        inputs, labels = self.mask_tokens(batch, self.remove_head)

        # attention mask at the character level, letting the mask characters have attention mask 1 at the word level
        attention_mask = torch.full(torch.Size([labels.shape[0], labels.shape[1]]), 1.0)

        return {"input_ids": inputs, "labels": labels, "attention_mask": attention_mask}


    def mask_tokens(self, inputs: List[torch.Tensor], remove_head) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()

        # We sample either the head or the tail in each sentence for mlm
        # i.e. masking is at the level of a word
        probability_matrix = torch.full(torch.Size([labels.shape[0], labels.shape[1]]), 0.0)
        for i in range(labels.shape[0]):
            if remove_head: probability_matrix[i][0] = 1
            else: probability_matrix[i][2] = 1

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        indices_replaced = (labels != -100)  

        inputs[indices_replaced] = self.entity_rel_dict["MASK"]

       
        return inputs, labels

def get_dicts(train_data_file, eval_data_file, test_data_file):
    file = open(train_data_file)
    lines = file.readlines()
    file.close()
    
    file = open(eval_data_file)
    lines += file.readlines()
    file.close()
    
    file = open(test_data_file)
    lines += file.readlines()
    file.close()
    
    lines = [line.strip().replace("\n", "") for line in lines]
    entity_rel_dict = {}
    
    for line in lines:
        head, rel, tail = line.split("\t")[0], line.split("\t")[1], line.split("\t")[2]
        if head not in entity_rel_dict:
            entity_rel_dict[head] = len(entity_rel_dict)
        if tail not in entity_rel_dict:
            entity_rel_dict[tail] = len(entity_rel_dict)
        if rel not in entity_rel_dict:
            entity_rel_dict[rel] = len(entity_rel_dict)
    
    entity_rel_dict["MASK"] = len(entity_rel_dict)
    
    dict_entity_rel = {}
    for key,value in entity_rel_dict.items():
        dict_entity_rel[value] = key
    
    return entity_rel_dict, dict_entity_rel

def get_datasets(train_data_file, eval_data_file, test_data_file):
    file = open(train_data_file)
    train_lines = file.readlines()
    file.close()
    
    file = open(eval_data_file)
    eval_lines = file.readlines()
    file.close()
    
    file = open(test_data_file)
    test_lines = file.readlines()
    file.close()
    
    train_lines = [line.strip().replace("\n", "") for line in train_lines]
    eval_lines = [line.strip().replace("\n", "") for line in eval_lines]
    test_lines = [line.strip().replace("\n", "") for line in test_lines]
    
    train_dataset = Dataset.from_dict({"text": train_lines})
    eval_dataset = Dataset.from_dict({"text": eval_lines})
    test_dataset = Dataset.from_dict({"text": test_lines})
    
    return train_dataset, eval_dataset, test_dataset

set_seed(42)

# building vocabulary from entities and relations
task = "fb15k237"
entity_rel_dict, dict_entity_rel = get_dicts("./data/{}/text/train.txt".format(task), "./data/{}/text/valid.txt".format(task), "./data/{}/text/test.txt".format(task))
train_dataset, eval_dataset, test_dataset = get_datasets("./data/{}/text/train.txt".format(task), "./data/{}/text/valid.txt".format(task), "./data/{}/text/test.txt".format(task))

def preprocess_function(example):
    inputs_ids = []
    splits = example["text"].split("\t")
    
    inputs_ids = [entity_rel_dict[splits[0]], entity_rel_dict[splits[1]], entity_rel_dict[splits[2]]]
    return {"input_ids": inputs_ids}

eval_dataset = eval_dataset.map(preprocess_function, num_proc=16)
test_dataset = test_dataset.map(preprocess_function, num_proc=16)

print(eval_dataset, len(eval_dataset["input_ids"][0]))

data_collator_rh = DataCollatorForLanguageModeling(entity_rel_dict, True)
data_collator_rt = DataCollatorForLanguageModeling(entity_rel_dict, False)
valid_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = 1, collate_fn=data_collator_rt, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, collate_fn=data_collator_rh, shuffle=False)

ntokens = len(entity_rel_dict)  # size of vocabulary
emsize = 256  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 5  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 8  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

device = torch.device("cuda")
model = Custom_model(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model.load_state_dict(torch.load("./model/m4mlm_fb15k237/checkpoint-26000/pytorch_model.bin"))

model = model.to(device)
model.eval()

labels, logits = [], []
for data in tqdm(test_loader):
    out = model(data["input_ids"].to(device), data["labels"].to(device), data["attention_mask"].to(device))
    logits.append(out["logits"].detach().cpu().numpy()[0])
    labels.append(data["labels"].detach().numpy()[0])

print(compute_metrics_func(np.array(logits), np.array(labels)))