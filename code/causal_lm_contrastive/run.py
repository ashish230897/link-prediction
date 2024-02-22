
from model import GPT
import os 
import json
import torch
from torch.utils.data import Dataset
from trainer import Trainer
import random

data = '../data/fb15k237/'
REVERSE=False

def create_vocab():
    words_2_id = {}
    id_2_words = {}
    id = 0
    words_2_id['UNK'] = 0
    id_2_words[0] = 'UNK'
    id+=1
    for file in ['train','valid','test']:
        with open(data+'text/'+file+'.txt') as f:
            temp = f.readlines()
            for row in temp:
                row = row.strip()
                n1, r, n2 = row.split()
                if n1 not in words_2_id:
                    words_2_id[n1] = id 
                    id_2_words[id] = n1
                    id+=1
                if n2 not in words_2_id:
                    words_2_id[n2] = id 
                    id_2_words[id] = n2
                    id+=1
                if r not in words_2_id:
                    words_2_id[r] = id 
                    id_2_words[id] = r
                    id+=1
    dataname = data.split('/')[2]
    with open(f'{dataname}-words2id.json', "w") as outfile: 
        json.dump(words_2_id, outfile)
    with open(f'{dataname}-id2words.json', "w") as outfile: 
        json.dump(id_2_words, outfile)
    return words_2_id,id_2_words

words2id, id2words = create_vocab()

print('Len of vocab', len(words2id))    

class CustomDataset(Dataset):
    def __init__(self, items, negative=1):
        if REVERSE:
            items.reverse()
        positives = [item[2:] for item in items]
        negatives = [list(set(random.sample(list(id2words.keys()),negative*2))-set(item))[:negative] for item in positives]
        self.inputs = []
        self.labels = []
        for i in range(len(items)):
            self.inputs.append(items[i])
            self.inputs.extend(items[i][:2]+[neg] for neg in negatives[i])
            self.labels.append(1)
            self.labels.extend([0]*negative)
        self.inputs = torch.tensor(self.inputs).to(torch.long)
        self.labels = torch.tensor(self.labels).to(torch.float)
        print(len(items),len(self.inputs),len(self.labels))

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        x = self.inputs[index,:]
        y = self.labels[index]
        return x,y


def create_dataset():
    dataset = {
        'train': [],
        'valid': [],
        'test': []
    }
    for file in ['train','valid','test']:
        with open(data+'text/'+file+'.txt') as f:
            temp = f.readlines()
            for row in temp:
                row = row.strip()
                n1, r, n2 = row.split()
                dataset[file].append([words2id[n1],words2id[r],words2id[n2]])

    train_dataset = CustomDataset(dataset['train'])
    val_dataset = CustomDataset(dataset['valid'])
    test_dataset = CustomDataset(dataset['test'])
    return train_dataset,val_dataset,test_dataset

train_dataset,val_dataset,test_dataset = create_dataset()

print('Train dataset len',len(train_dataset))
print('Val dataset len',len(val_dataset))
print('Test dataset len',len(test_dataset))


model_config = GPT.get_default_config()
model_config.model_type = 'gpt-tiny'
model_config.vocab_size = len(words2id) # openai's model vocabulary
model_config.block_size = 3  # openai's model block_size (i.e. input context length)

model = GPT(model_config)
print(model_config)


train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-6 # many possible options, see the file
train_config.max_iters = 7000
train_config.batch_size = 32
train_config.weight_decay = 1e-7
trainer = Trainer(train_config, model, train_dataset,val_dataset,test_dataset,train_every=100,val_every=500)
trainer.run()

torch.save(model, 'gpt-tiny.pt')