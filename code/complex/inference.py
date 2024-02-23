import torch
import sys 
import torch
from torch_geometric.datasets import FB15k_237, WordNet18RR
from tqdm import tqdm
from complex import ComplEx


dataset_name = sys.argv[1]
assert dataset_name in ["fb15k_237", "wordnet18rr"]

device = 'cuda'

def get_data_complex(data):
    if data == "fb15k_237":
        train_data = FB15k_237("../../data/fb15k237/", split='train')[0].to(device)
        val_data = FB15k_237("../../data/fb15k237/", split='val')[0].to(device)
        test_data = FB15k_237("../../data/fb15k237/", split='test')[0].to(device)
    elif data == "wordnet18rr":
        train_data = FB15k_237("../../data/wordnet18rr/", split='train')[0].to(device)
        val_data = FB15k_237("../../data/wordnet18rr/", split='val')[0].to(device)
        test_data = FB15k_237("../../data/wordnet18rr/", split='test')[0].to(device)
    
    return train_data, val_data, test_data

@torch.no_grad()
def evaluate_complex():
    _, val_data, test_data = get_data_complex(dataset_name)
    complex = torch.load(f'../../model/complex-{dataset_name}.pt').to('cuda')

    rank, mrr, hits_at_k, hits_at_one, mAP = complex.test(
        head_index=val_data.edge_index[0],
        rel_type=val_data.edge_type,
        tail_index=val_data.edge_index[1],
        batch_size=50000,
        k=10,
    )
    print(f'val Mean Rank: {rank:.2f}',
              f'val MRR: {mrr:.4f}', 
              f'val Hits@10: {hits_at_k:.4f}',
              f'val Hits@1: {hits_at_one:.4f}',
              f'val mAP@10: {mAP:.4f}')
    rank, mrr, hits_at_k, hits_at_one, mAP = complex.test(
        head_index=test_data.edge_index[0],
        rel_type=test_data.edge_type,
        tail_index=test_data.edge_index[1],
        batch_size=50000,
        k=10,
    )
    print(f'test Mean Rank: {rank:.2f}',
              f'test MRR: {mrr:.4f}', 
              f'test Hits@10: {hits_at_k:.4f}',
              f'test Hits@1: {hits_at_one:.4f}',
              f'test mAP@10: {mAP:.4f}')


if __name__=='__main__':
    evaluate_complex()


