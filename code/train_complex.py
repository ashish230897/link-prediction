import torch
import torch.optim as optim

from torch_geometric.datasets import FB15k_237, WordNet18RR
from torch_geometric.nn import ComplEx

device = 'cuda'

def get_data(data):
    
    if data == "fb15k_237":
        train_data = FB15k_237("./data/fb15k237/", split='train')[0].to(device)
        val_data = FB15k_237("./data/fb15k237/", split='val')[0].to(device)
        test_data = FB15k_237("./data/fb15k237/", split='test')[0].to(device)
    elif data == "wordnet18rr":
        train_data = FB15k_237("./data/wordnet18rr/", split='train')[0].to(device)
        val_data = FB15k_237("./data/wordnet18rr/", split='val')[0].to(device)
        test_data = FB15k_237("./data/wordnet18rr/", split='test')[0].to(device)
    
    return train_data, val_data, test_data

train_data, val_data, test_data = get_data("wordnet18rr")

print("Train data num of nodes: ", train_data.num_nodes)
print("Train data num of edge types: ", train_data.num_edge_types)

model = ComplEx(
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=50,
).to(device)


loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
)

print(model)
print(loader)

optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6)

def train():
    model.train()
    total_loss = total_examples = 0
    
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
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
        batch_size=20000,
        k=10,
    )


for epoch in range(1, 501):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    if epoch % 25 == 0:
        rank, mrr, hits = test(val_data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')


rank, mrr, hits_at_10 = test(test_data)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')
    


