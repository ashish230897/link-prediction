import wandb
import logging
import os
import random
import torch
from torch import Tensor, nn
import numpy as np
import transformers.models.roformer.modeling_roformer as modeling_roformer
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, NewType, Tuple
from datasets import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    SchedulerType,
    IntervalStrategy
)
import math

# os.environ["WANDB_DISABLED"] = "true"
tokenizer = None
count = 1
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    
    test_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dropout_rate: Optional[float] = field(
        default=0.1,
        metadata={"help": "specify dropout rate"},
    )
    
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project name"},
    )
    
    saved_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )


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
    def __init__(self, entity_rel_dict):
        self.entity_rel_dict = entity_rel_dict

    # def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    def __call__(self, examples) -> Dict[str, torch.Tensor]:
		
        # now we have to pad all sentences to max length and also truncate longer ones
		# first truncate
        tokenized_lines = [example['input_ids'] for example in examples]

        # deal with masks
        
        # cls, sep1, sep2, relation words, end wont be masked
        lines_masks = [ ['NOMASK', 'MASK', 'NOMASK', 'NOMASK', 'NOMASK', 'MASK', 'NOMASK'] for _ in tokenized_lines]
        masks = np.full(np.shape(lines_masks), "MASK")
        tomasks = masks == lines_masks

        tokenized_lines = np.array(tokenized_lines, dtype=int)

        # tensorizing
        tokenized_lines = [torch.tensor(line) for line in tokenized_lines]
        tomask = [torch.tensor(line) for line in tomasks]
        batch = torch.stack(tokenized_lines, dim=0)
        tomask = torch.stack(tomask, dim=0)

        inputs, labels = self.mask_tokens([batch, tomask])

        # attention mask at the character level, letting the mask characters have attention mask 1 at the word level
        attention_mask = torch.full(torch.Size([labels.shape[0], labels.shape[1]]), 1.0)

        return {"input_ids": inputs, "labels": labels, "attention_mask": attention_mask}


    def mask_tokens(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        [inputs, tomask] = inputs
        labels = inputs.clone()

        # We sample either the head or the tail in each sentence for mlm
        # i.e. masking is at the level of a word
        probability_matrix = torch.full(torch.Size([labels.shape[0], labels.shape[1]]), 0.0)
        for i in range(labels.shape[0]):
            probability_matrix[i][random.choice([1,5])] = 1

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        indices_replaced = (labels != -100)  

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        word_indices_replaced_wmask = torch.bernoulli(torch.full(masked_indices.shape, 0.8)).bool() & masked_indices
        mask_replace_input_indices = indices_replaced*word_indices_replaced_wmask
        inputs[mask_replace_input_indices] = self.entity_rel_dict["MASK"]

        # 10% of the time, we replace masked input tokens with random word
        word_indices_random = torch.bernoulli(torch.full(masked_indices.shape, 0.5)).bool() & masked_indices & ~word_indices_replaced_wmask
        random_words = torch.randint(len(self.entity_rel_dict), labels.shape, dtype=torch.long)
        inputs[word_indices_random] = random_words[word_indices_random]

        del tomask
       
        return inputs, labels

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 7):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # batch_size, seq_length, hidden_dim
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x)


class Custom_model(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.ntoken = ntoken

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_ids, labels, attention_mask):
        # logger.info(f'Input ids : {input_ids.shape}')
        # logger.info(input_ids)

        # shape: (batch_size, sentence_size, word_size, hidden_size)
        input_embeds = self.embedding(input_ids) * math.sqrt(self.d_model)
        input_embeds = self.pos_encoder(input_embeds)
        #print("input embeds size: ", input_embeds.size())
        
        output = self.transformer_encoder(input_embeds)
        output = self.linear(output)
        #print("output size: ", output.size())
        
        outputs = {}

        outputs['logits'] = output
        #outputs['labels'] = input_ids

        logits_flat = output.view(-1, self.ntoken)
        labels_flat = labels.view(-1)

        criterion = nn.CrossEntropyLoss()
        outputs["loss"] = criterion(logits_flat, labels_flat)  # masked word modeling loss
        return outputs


def get_dicts(train_data_file, eval_data_file):
    file = open(train_data_file)
    lines = file.readlines()
    file.close()
    
    file = open(eval_data_file)
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
    
    # special tokens
    entity_rel_dict["SEP1"] = len(entity_rel_dict)
    entity_rel_dict["SEP2"] = len(entity_rel_dict)
    entity_rel_dict["CLS"] = len(entity_rel_dict)
    entity_rel_dict["END"] = len(entity_rel_dict)
    entity_rel_dict["MASK"] = len(entity_rel_dict)
    
    dict_entity_rel = {}
    for key,value in entity_rel_dict.items():
        dict_entity_rel[value] = key
    
    return entity_rel_dict, dict_entity_rel

def get_datasets(train_data_file, eval_data_file):
    file = open(train_data_file)
    train_lines = file.readlines()
    file.close()
    
    file = open(eval_data_file)
    eval_lines = file.readlines()
    file.close()
    
    train_lines = [line.strip().replace("\n", "") for line in train_lines]
    eval_lines = [line.strip().replace("\n", "") for line in eval_lines]
    
    train_dataset = Dataset.from_dict({"text": train_lines})
    eval_dataset = Dataset.from_dict({"text": eval_lines})
    
    return train_dataset, eval_dataset

def compute_metrics_func(eval_preds):
    logits, labels = eval_preds
    indices = np.where(labels != -100)
    logits = logits[indices]
    labels = labels[indices]
    top_predictions = np.argsort(logits, axis=1)[:, ::-1] 

    hits_at_1 = np.sum(top_predictions[:, 0] == labels)/len(labels)
    hits_at_10 = 0
    for i in range(len(labels)):
        if labels[i] in top_predictions[i][:10]:
            hits_at_10 += 1
    hits_at_10 = hits_at_10 / len(labels)

    reciprocal_ranks = []
    for i in range(len(labels)):
        rank = np.where(top_predictions[i] == labels[i])[0]
        if len(rank) > 0:
            reciprocal_rank = 1.0 / (rank[0] + 1)
            reciprocal_ranks.append(reciprocal_rank)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    average_precisions = []
    for i in range(len(labels)):
        relevant_indices = np.where(top_predictions[i] == labels[i])[0]
        if len(relevant_indices) > 0:
            precision_at_k = []
            for j in range(1, len(relevant_indices) + 1):
                precision_at_k.append(np.sum(labels[i] == top_predictions[i][:j]) / j)
            average_precisions.append(np.mean(precision_at_k))
    map_ = np.mean(average_precisions) if average_precisions else 0.0

    return {"hits@1": hits_at_1, "hits@10": hits_at_10, "MRR": mrr, "MAP": map_}

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set seed
    set_seed(training_args.seed)
    # set the schedular type
    training_args.lr_scheduler_type = SchedulerType.LINEAR
    training_args.evaluation_strategy = IntervalStrategy.STEPS
    training_args.logging_strategy = IntervalStrategy.STEPS
    training_args.save_strategy = IntervalStrategy.STEPS

    # building vocabulary from entities and relations
    entity_rel_dict, dict_entity_rel = get_dicts(data_args.train_data_file, data_args.eval_data_file)
    train_dataset, eval_dataset = get_datasets(data_args.train_data_file, data_args.eval_data_file)
    
    def preprocess_function(example):
        inputs_ids = []
        splits = example["text"].split("\t")
        
        inputs_ids = [entity_rel_dict["CLS"], entity_rel_dict[splits[0]], entity_rel_dict["SEP1"], entity_rel_dict[splits[1]], entity_rel_dict["SEP2"], 
                      entity_rel_dict[splits[2]], entity_rel_dict["END"]]
        return {"input_ids": inputs_ids}

    train_dataset = train_dataset.shuffle(seed=training_args.seed).map(preprocess_function, num_proc=16)
    eval_dataset = eval_dataset.map(preprocess_function, num_proc=16)

    print(train_dataset, len(train_dataset["input_ids"][0]))
    print(train_dataset["input_ids"][0])
    print(train_dataset["input_ids"][1])
    
    data_collator = DataCollatorForLanguageModeling(entity_rel_dict)
    # loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, collate_fn=data_collator)
    
    ntokens = len(entity_rel_dict)  # size of vocabulary
    emsize = 128  # embedding dimension
    d_hid = 300  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 3  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 8  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.1  # dropout probability
    device = torch.device("cuda")
    model = Custom_model(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    wandb.init(
        # set the wandb project where this run will be logged
        project=data_args.wandb_project,
        # track hyperparameters and run metadata
        config={
            "learning_rate": training_args.learning_rate,
            "architecture": "Transformer",
            "steps": training_args.max_steps
        }
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters are:--------------- ", pytorch_total_params)
    
    print("Trainer args:--------------- ")
    print(training_args)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_func,
    )

    # Training
    model_path = (
        model_args.model_name_or_path
        if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
        else None
    )
    trainer.train(model_path=model_path)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        # save the best model
        trainer.save_model(os.path.join(training_args.output_dir,'final_model'))
        



if __name__ == "__main__":
    main()