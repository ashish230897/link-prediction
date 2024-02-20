import os 

data = 'fb15k237/'

nodes = {
    'train':[],
    'valid':[],
    'test':[]
}

for file in ['train','valid','test']:
    with open(data+'text/'+file+'.txt') as f:
        content = f.readlines()
    for row in content:
        row = row.strip()
        n1,r,n2 = row.split()
        nodes[file].append(n1)
        nodes[file].append(n2)
    nodes[file] = set(nodes[file])

print('Data = ',data)
overlap_train_valid = nodes['train'].intersection(nodes['valid'])
overlap_size = len(overlap_train_valid)
valid_size = len(nodes['valid'])
non_overlap_size = valid_size - overlap_size
percentage_non_overlap = (non_overlap_size / valid_size) * 100
print("Non overlap percent of valid:", percentage_non_overlap)

overlap_train_test = nodes['train'].intersection(nodes['test'])
overlap_size = len(overlap_train_test)
test_size = len(nodes['test'])
non_overlap_size = test_size - overlap_size
percentage_non_overlap = (non_overlap_size / test_size) * 100
print("Non overlap percent of test:", percentage_non_overlap)


data = 'wordnet18rr/'

nodes = {
    'train':[],
    'valid':[],
    'test':[]
}


for file in ['train','valid','test']:
    with open(data+'text/'+file+'.txt') as f:
        content = f.readlines()
    for row in content:
        row = row.strip()
        n1,r,n2 = row.split()
        nodes[file].append(n1)
        nodes[file].append(n2)
    nodes[file] = set(nodes[file])

print('Data = ',data)
overlap_train_valid = nodes['train'].intersection(nodes['valid'])
overlap_size = len(overlap_train_valid)
valid_size = len(nodes['valid'])
non_overlap_size = valid_size - overlap_size
percentage_non_overlap = (non_overlap_size / valid_size) * 100
print("Non overlap percent of valid:", percentage_non_overlap)

overlap_train_test = nodes['train'].intersection(nodes['test'])
overlap_size = len(overlap_train_test)
test_size = len(nodes['test'])
non_overlap_size = test_size - overlap_size
percentage_non_overlap = (non_overlap_size / test_size) * 100
print("Non overlap percent of test:", percentage_non_overlap)