import torch
import torch.optim as optim

from torch_geometric.datasets import FB15k_237, WordNet18RR
#from torch_geometric.nn import ComplEx
from complex import ComplEx
from tqdm import tqdm
import sys 

device = 'cuda'

def get_data(data):
    
    if data == "fb15k_237":
        train_data = FB15k_237("../../data/fb15k237/", split='train')[0].to(device)
        val_data = FB15k_237("../../data/fb15k237/", split='val')[0].to(device)
        test_data = FB15k_237("../../data/fb15k237/", split='test')[0].to(device)
    elif data == "wordnet18rr":
<<<<<<< HEAD:code/train_complex.py
        train_data = WordNet18RR("./data/wordnet18rr/", split='train')[0].to(device)
        val_data = WordNet18RR("./data/wordnet18rr/", split='val')[0].to(device)
        test_data = WordNet18RR("./data/wordnet18rr/", split='test')[0].to(device)
=======
        train_data = FB15k_237("../../data/wordnet18rr/", split='train')[0].to(device)
        val_data = FB15k_237("../../data/wordnet18rr/", split='val')[0].to(device)
        test_data = FB15k_237("../../data/wordnet18rr/", split='test')[0].to(device)
>>>>>>> new-mlm:code/complex/train_complex.py
    
    return train_data, val_data, test_data

dataset_name = sys.argv[1]
assert dataset_name in ["fb15k_237", "wordnet18rr"]

train_data, val_data, test_data = get_data(dataset_name)

print("Train data num of nodes: ", train_data.num_nodes)
print("Train data num of edge types: ", train_data.num_edge_types)

model = ComplEx(
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
    sparse=False
).to(device)


val_loader = model.loader(
    head_index=val_data.edge_index[0],
    rel_type=val_data.edge_type,
    tail_index=val_data.edge_index[1],
    batch_size=10000,
    shuffle=True,
)

loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=10000,
    shuffle=True,
)


print(model)
print(loader)


optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)

def train():
    model.train()
    total_loss = total_examples = 0
<<<<<<< HEAD:code/train_complex.py
    
    for head_index, rel_type, tail_index in loader:
        # print()
=======
    for head_index, rel_type, tail_index in (loader):
>>>>>>> new-mlm:code/complex/train_complex.py
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    
    return total_loss / total_examples

@torch.no_grad()
def eval():
    model.eval()
    total_loss = total_examples = 0
    
    for head_index, rel_type, tail_index in (val_loader):
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples

@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=50000,
        k=10,
    )

# a,b = test(val_data)
# print(a,b)
# rank, mrr, hits_at_k, hits_at_one, mAP = test(val_data)
# print(rank,mrr,hits_at_k,hits_at_one,mAP)

for epoch in range(1, 15):
    loss = train()
    val_loss = eval()
    print(f'Epoch: {epoch:03d},Train Loss: {loss:.4f},Val Loss: {val_loss:.4f}' )
    
    if epoch % 5 == 0:
        rank, mrr, hits_at_k, hits_at_one, mAP = test(val_data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}',
              f'Val MRR: {mrr:.4f}', 
              f'Val Hits@10: {hits_at_k:.4f}',
              f'Val Hits@1: {hits_at_one:.4f}',
              f'Val mAP@10: {mAP:.4f}')


rank, mrr, hits_at_k, hits_at_one, mAP = test(test_data)
print(f'Test Mean Rank: {rank:.2f}', 
      f'Test MRR: {mrr:.4f}',
      f'Test Hits@10: {hits_at_k:.4f}',
      f'Test Hits@1: {hits_at_one:.4f}',
      f'Test mAP@10: {mAP:.4f}')
    
print(f'Saving model to ../../model/complex-{dataset_name}.pt')
torch.save(model,f'../../model/complex-{dataset_name}.pt')
