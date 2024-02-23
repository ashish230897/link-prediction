import torch 
import sys 
from model import GPT
import os 
import json
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from eval_func import compute_metrics_func
from tqdm import tqdm
import random

dataset_name = sys.argv[1]
assert dataset_name in ["fb15k237", "wordnet18rr"]
data = f'../../data/{dataset_name}/'
MODE=sys.argv[2]

print('Dataset ',dataset_name)
print('MODE ',MODE)

def create_vocab():
    words_2_id = {}
    id_2_words = {}
    id = 0
    words_2_id['UNK'] = 0
    id_2_words[0] = 'UNK'
    id+=1
    words_2_id['MASK'] = 1
    id_2_words[1] = 'MASK'
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

class CustomDataset(Dataset):
    def __init__(self, items):
        if MODE=='subject':
            self.inputs = [[1]+item[1:] for item in items]
            self.labels = [item[:1]+[-100,-100] for item in items]
        elif MODE=='object':
            self.inputs = [item[:2]+[1] for item in items]
            self.labels = [[-100,-100]+item[2:] for item in items]
        elif MODE=='random':
            idx = [random.randint(1, 2) for _ in range(len(items))]
            self.inputs = []
            self.labels = []
            for i in range(len(items)):
                if idx[i] == 1:
                    self.inputs.append([1]+items[i][1:])
                    self.labels.append(items[i][:1]+[-100,-100])
                else:
                    self.inputs.append(items[i][:2]+[1])
                    self.labels.append([-100,-100]+items[i][2:])
        self.inputs = torch.tensor(self.inputs).to(torch.long)
        self.labels = torch.tensor(self.labels).to(torch.long)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        x = self.inputs[index,:]
        y = self.labels[index,:]
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

model = torch.load(f'../../model/gpt-tiny-mlm-{dataset_name}-{sys.argv[2]}.pt').to('cuda')

print('STARTING VALIDATION--------------')
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    pin_memory=True,
    batch_size=1024,
    num_workers=4,
)
val_loss = []
logits_store = None 
labels_store = None  
for batch in tqdm(val_loader):
    batch = [t.to('cuda') for t in batch]
    x, y = batch
    if labels_store==None:
        labels_store = y
    else: 
        labels_store = torch.cat([labels_store,y],dim=0)
    with torch.no_grad():
        logits, val_batch_loss = model(x, y)
        if logits_store==None:
            logits_store = logits
        else: 
            logits_store = torch.cat([logits_store,logits],dim=0)
        val_loss.append(val_batch_loss)
#print(logits_store.shape,labels_store.shape)
print(compute_metrics_func(logits_store.detach().cpu().numpy(),labels_store.detach().cpu().numpy()))
#print('GPU TO CPU')
print('Val loss',(sum(val_loss)/len(val_loss)).item())

print('STARTING TEST--------------')
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    pin_memory=True,
    batch_size=1024,
    num_workers=4,
)
test_loss = []
logits_store = None 
labels_store = None  
for batch in tqdm(test_loader):
    batch = [t.to('cuda') for t in batch]
    x, y = batch
    if labels_store==None:
        labels_store = y
    else: 
        labels_store = torch.cat([labels_store,y],dim=0)
    with torch.no_grad():
        logits, test_batch_loss = model(x, y)
        if logits_store==None:
            logits_store = logits
        else: 
            logits_store = torch.cat([logits_store,logits],dim=0)
        test_loss.append(val_batch_loss)
#print(logits_store.shape,labels_store.shape)
print(compute_metrics_func(logits_store.detach().cpu().numpy(),labels_store.detach().cpu().numpy()))
#print('GPU TO CPU')
print('Test loss',(sum(test_loss)/len(test_loss)).item())
