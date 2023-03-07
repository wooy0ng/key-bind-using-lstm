import torch
from torch.utils.data import DataLoader, TensorDataset
import time

from load_datasets import load_datasets
from utils import *
from model import *
from save_obj import *

'''
    [mode.py]
    train / test / key_train / key_test mode 별로 구성된 func 제공
'''

PREFIX = "\n[Actual Label = {}, Predicted Label = {}]\nMaster Key \t: 0x{}\nPredicted Key \t: 0x{}\nSimilarity = {}%"

# classfication model train func
def train(args):
    # load datasets
    datasets = load_datasets(args)

    data_loader = DataLoader(
        datasets, 
        batch_size=args.batch_size, 
        shuffle=True
    )

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] {device} is available")
    
    # load model
    model = LSTMClassification(
        in_size=datasets.size(2),
        hidden_size=args.hidden_size,
        out_size=1,
        sequence_length=datasets.size(1),
        num_layers=args.num_layers,
        device=device
    ).to(device)

    if args.pretrained_mode is True:
        print("[+] pretrained mode")
        model = load_model(args, model)

    # model train
    start = time.time()
    for epoch in range(args.epoch):
        model.losses = []
        for data, labels in data_loader:
            model.train(data, labels)

        if epoch % 10 == 0:
            loss = sum(model.losses) / len(data_loader)
            t = time.time() - start
            print(f'[{epoch+1} epoch, {model.iteration} iteration] mean loss : {loss:.3f}\t({t:.3f} sec)')
            start = time.time()
    try:
        save_model(args, model, 'model')   # save model
        save_object(model.context, 'context_vector')  # save context_vector
    except BaseException as e:
        print(e)

# classfication model test func
def test(args):
    args.mode = 'test'
    datasets = load_datasets(args)

    data_loader = DataLoader(
        datasets, 
        batch_size=1, 
        shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] {device} is available")
    
    # load clf model
    model = LSTMClassification(
        in_size=datasets.size(2),
        hidden_size=args.hidden_size,
        out_size=1,
        sequence_length=datasets.size(1),
        num_layers=args.num_layers,
        device=device
    ).to(device)

    model = load_model(args, model, 'model')
    
    # model test
    files_name = datasets.files_name
    correct = 0
    with torch.no_grad():
        for file_name, (data, labels) in zip(files_name, data_loader):
            inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            predicted = model(inputs)
            predicted = round(predicted.item())
            # loss = criterion(predicted, labels)
            print(f"{file_name} predicted labels : {predicted}\tactual labels : {labels.item()}")
            if predicted == labels.item():
                correct += 1
    
    torch.set_printoptions(precision=8, sci_mode=False)
    # save context vector
    # print(f"context vector : {model.context}")
    # print(f"derivative : {F_Tanh_derivative(model.context)}")    
    print(f"accuracy : {correct / len(files_name) * 100:.1f}%")
    
# key generation train func
def key_train(args):
    args.mode = 'train'
    datasets = load_datasets(args)

    data_loader = DataLoader(
        datasets, 
        batch_size=args.batch_size, 
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load clf model
    model = LSTMClassification(
        in_size=datasets.size(2),
        hidden_size=args.hidden_size,
        out_size=1,
        sequence_length=datasets.size(1),
        num_layers=args.num_layers,
        device=device
    ).to(device)

    model = load_model(args, model, 'model')

    X, y = [], []
    with torch.no_grad():
        for idx, (data, labels) in enumerate(data_loader):
            inputs = data
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device)

            predicted = model.forward(inputs)
            predicted = (predicted > 0.5).float()
            for idx, p in enumerate(predicted):
                # p == 1
                if (p and labels[idx]) and (p):
                    X.append(model.context)
                    y.append(torch.round(predicted))
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    train_set = TensorDataset(X, y)
    
    data_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True
    ) ### end load clf model

    # random key generation
    key_generated_class = MakeRandomKey(args)
    generated_key = key_generated_class()

    key_similarity_model = KeyGeneration(args).to(device)
    key = key_generated_class.key.view(1, -1).to(device)

    # model train
    start = time.time()
    for epoch in range(args.key_epoch):
        key_similarity_model.losses = []
        for data, labels in data_loader:
            #print(f'data : {data}, label : {labels}')
            key_similarity_model.train(data, labels, key)
            
        if epoch % 10 == 0:
            loss = sum(key_similarity_model.losses) / len(data_loader)
            t = time.time() - start
            print(f'[{epoch+1} epoch, {key_similarity_model.iteration} iteration] key mean loss : {loss:.3f}\t({t:.3f} sec)')
            start = time.time()        
    try:
        save_model(args, key_similarity_model, 'key_similarity_model')   # save model
        # save_object(model.context, 'context_vector')  # save context_vector
    except BaseException as e:
        print(e)

# key generation test func
def key_test(args):
    args.mode = 'test'
    datasets = load_datasets(args)

    data_loader = DataLoader(
        datasets, 
        batch_size=1, 
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] {device} is available")

    model = LSTMClassification(
        in_size=datasets.size(2),
        hidden_size=args.hidden_size,
        out_size=1,
        sequence_length=datasets.size(1),
        num_layers=args.num_layers,
        device=device
    ).to(device)

    # load key-gen model and key
    model = load_model(args, model, 'model')
    key_generated_class = MakeRandomKey(args)
    generated_key = key_generated_class()

    key_similarity_model = KeyGeneration(args).to(device)
    key_similarity_model = load_model(args, key_similarity_model, 'key_similarity_model')
    key = key_generated_class.key.view(1, -1).to(device)
    
    files_name = datasets.files_name
    correct = 0
    
    # model test
    logs = {}
    idx = 0
    with torch.no_grad():  
        for file_name, (data, labels) in zip(files_name, data_loader):
            inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            predicted = model(inputs)
            predicted = (predicted > 0.5).float()

            predicted_key = key_similarity_model.test(model.context, predicted, device)
            predicted_key = (predicted_key.squeeze() > 0.5).float()
            torch.set_printoptions(precision=3, sci_mode=False)
            
            # print results
            print(PREFIX.format(labels.item(), predicted.item(), bit_to_hex(key), bit_to_hex(predicted_key), bit_similarity(predicted_key, key)))

            if predicted == labels.item():
                correct += 1

            # print results 2
            log = {}
            log['predicted_key'] = bit_to_hex(predicted_key)
            log['key'] = bit_to_hex(key)
            log['file_name'] = file_name
            log['predicted_label'] = predicted.item()
            log['actual_label'] = labels.item()
            log['similarity'] = bit_similarity(predicted_key, key)

            logs[idx] = log
            idx += 1

    return logs

def evaluate(args):
    args.mode = "evaluate"
    datasets = load_datasets(args)
    
    data_loader = DataLoader(
        datasets, 
        batch_size=1, 
        shuffle=False
    )
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"[+] {device} is available")

    # load model
    model = LSTMClassification(
        in_size=datasets.size(2),
        hidden_size=args.hidden_size,
        out_size=1,
        sequence_length=datasets.size(1),
        num_layers=args.num_layers,
        device=device
    ).to(device)

    model = load_model(args, model, 'model')

    key_generated_class = MakeRandomKey(args)
    generated_key = key_generated_class()

    key_similarity_model = KeyGeneration(args).to(device)
    key_similarity_model = load_model(args, key_similarity_model, 'key_similarity_model')
    key = key_generated_class.key.view(1, -1).to(device)

    files_name = datasets.files_name
    correct = 0
    torch.set_printoptions(precision=8, sci_mode=False)
    with torch.no_grad():
        
        iterate = iter(data_loader)
        data = next(iterate)

        inputs = data[0].to(device)
        predicted = model(inputs)
        predicted = round(predicted.item())
        print(f"predicted labels : {predicted}")

        predicted = torch.tensor(predicted).view(-1, 1)
        predicted_key = key_similarity_model.test(model.context, predicted)
        predicted_key = (predicted_key > 0.5).float()
        #log = {'predicted_key' : bit_to_hex(predicted_key), 'key' : bit_to_hex(key), 'similarity': bit_similarity(predicted_key, key)}
        print(f"predicted key : \t{bit_to_string(predicted_key)}  ({bit_similarity(predicted_key, key)}%)\t(0x{bit_to_hex(predicted_key)})")
        print(f"actual key : \t\t{bit_to_string(key)}  (100.0%)\t(0x{bit_to_hex(key)})")

        # for file_name, (data, labels) in zip(files_name, data_loader):
        #     inputs = data
        #     inputs = inputs.to(device)
        #     # labels = labels.to(device)
            
        #     predicted = model(inputs)
        #     predicted = round(predicted.item())
        #     # loss = criterion(predicted, labels)
        #     print(f"[{file_name}] predicted labels : {predicted}")

        #     predicted = torch.tensor(predicted).view(-1, 1)
        #     predicted_key = key_similarity_model.test(model.context, predicted)
        #     predicted_key = (predicted_key > 0.5).float()

        #     print(f"predicted key : \t{bit_to_string(predicted_key)}  ({bit_similarity(predicted_key, key)}%)\t(0x{bit_to_hex(predicted_key)})")
        #     print(f"actual key : \t\t{bit_to_string(key)}  (100.0%)\t(0x{bit_to_hex(key)})")

    #return log
