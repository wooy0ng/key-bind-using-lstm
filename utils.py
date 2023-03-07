import os
import numpy as np
from sqlalchemy import false
import torch
import torch.nn as nn

# generate random master key
class MakeRandomKey(nn.Module):
    def __init__(self, args):
        super(MakeRandomKey, self).__init__()
        self.master_key_size = args.master_key_size
        self.batch_size = args.batch_size
        self.mode = args.mode
        self.master_key = args.master_key
        
    def __call__(self):
        if self.mode == 'key_train':
            # make random key
            result = []
            master_key = torch.randint(0, 2, (self.master_key_size, ))
            with open(self.master_key, 'w+') as f:
                f.write(bit_to_string(master_key))
        else:
            result = []
            with open(self.master_key, 'r') as f:
                master_key = f.readline()
            master_key = string_to_bit(master_key)

        master_key = master_key.to(dtype=torch.float32)
        self.key = master_key
        for _ in range(self.batch_size):
            result.append(master_key)
        result = torch.stack(result)
        
        return result
    
def F_Tanh_derivative(x):
    return (1 - (pow(x, 2)))
        
def devided_windows(args, x):
    import itertools
    x_seq = []
    windows = args.windows
    batch_size = x.shape[0]
    sequence_length = x.shape[1]

    for batch, seq in itertools.product(range(batch_size), range(sequence_length-windows)):
        x_seq.append([x[batch][seq:seq+windows]])

    x_seq = torch.tensor(x_seq)
    x_seq = torch.transpose(x_seq, 1, 0)
    return x_seq.to(dtype=torch.float32)

def get_file_names(PATH):
    import os
    result = []
    for file in os.listdir(PATH):
        result.append(os.path.join(PATH, file))
    return result

def bit_to_string(data):
    data = data.to(dtype=torch.int8)
    data = str(data.tolist()).lstrip('[').rstrip(']').split(', ')
    data = ''.join(data)
    return data

def string_to_bit(data):
    return torch.tensor([int(bit) for bit in data])

def separate_dir(args):
    dirs = os.listdir(args.FILEPATH)
    true_files, false_files = [], []
    for dir in dirs:
        if dir == args.set_true_label or dir == 'true':
            true_files.append(get_file_names(
                os.path.join(args.FILEPATH, dir)
            ))
        else:
            false_files.append(get_file_names(
                os.path.join(args.FILEPATH, dir)
            ))
    return true_files, false_files

def padding(arr, n):
    rows, cols = arr.shape
    rows_exponent = int(np.log2(rows))
    cols_exponent = int(np.log2(cols))

    result = arr.copy()
    if rows > (2**rows_exponent):
        cnt = (2**(rows_exponent + 1)) - rows
        for _ in range(cnt):
            stack = np.zeros(cols,).fill(n)
            stack.fill(n)
            result = np.vstack((result, stack))
    if cols > (2**cols_exponent):
        cnt = (2**(cols_exponent + 1)) - cols
        for _ in range(cnt):
            stack = np.zeros((rows, 1))
            stack.fill(n)
            result = np.hstack((result, stack))
    return result

def data_crop_or_padding(data, sr, time_limited):
    # padding
    if data.shape[0] < time_limited:
        size = data.shape[0]
        z = np.zeros(sr * (int(size / sr) + 1) - size)
        data = np.append(data, z)
    else:
        data = data[:time_limited]
    return data

def convert_to_key(predicted):
    result = []
    for bits in predicted:
        result.append([round(bit.item()) for bit in bits])
    return torch.tensor(result).to(dtype=torch.float32) 

def isTensorEqual(a, b):
    comp = torch.eq(a, b)
    if False not in comp:
        return True
    return False

def bit_similarity(a, b):
    comp = torch.eq(a, b).squeeze()
    cnt = 0
    for bool in comp:
        if bool:
            cnt += 1
    return (cnt / len(comp)) * 100

def bit_to_hex(data):
    return '%08X' % int(bit_to_string(data), 2)