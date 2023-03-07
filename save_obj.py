import torch
import pickle as pkl
import os

'''
    [save_obj.py]
    save model / load model (clf model 혹은 key-gen model 저장 시 사용)
    save object / load object (context vector 저장 시 사용)
'''

def save_model(args, model, name='obj'):
    print("\n[+] model save mode", end=' ')
    torch.save(model.state_dict(), os.path.join(args.model_path, name+'.pt'))
    print("[complete]")

def load_model(args, model, name='obj'):
    print("[+] model load mode", end=' ')
    model.load_state_dict(torch.load(os.path.join(args.model_path, name+'.pt')))
    print("[complete]\n")
    return model

def save_object(object, name='obj'):
    print("\n[+] object save mode", end=' ')
    pkl.dump(object, open(f"{name}.pkl", "wb+"))
    print("[complete]")

def load_object(args, name='obj'):
    print("[+] object load mode", end=' ')
    with open(f"{name}.pkl", "rb+") as obj:
        _object = pkl.load(obj)
    print("[complete]\n")
    return _object
    