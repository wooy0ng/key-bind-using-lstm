import json
import torch
import argparse

with open('./config.json', encoding="utf-8") as f:
    config = json.load(f)

torch.multiprocessing.set_sharing_strategy('file_system')

# Argument parser
def get_config():
    parser = argparse.ArgumentParser(add_help=False)

    ### path
    parser.add_argument('--path', required=False, default=config['path']['base'])
    parser.add_argument('--model_path', required=False, default=config['path']['model_base'])
    parser.add_argument('--train_path', type=str, default=config['path']['train'])
    parser.add_argument('--test_path', type=str, default=config['path']['test'])
    parser.add_argument('--val_path', type=str, default=config['path']['val'])

    ### model
    parser.add_argument('--use_gpu', action="store_true", default=config['model']['use_gpu'])
    
    parser.add_argument('--mode', type=str, default=config['model']['mode'], help='train, test, key_train, key_test')
    parser.add_argument('--pretrained_mode', required=False, default=config['model']['pre_mode'])
    parser.add_argument('--pretrained_model', required=False, default=config['model']['pre_trained_model'])

    parser.add_argument('--sample_rate', action="store_true", default=config['model']['sample_rate'])
    parser.add_argument('--duration', action="store_true", default=config['model']['duration'])

    parser.add_argument('--batch_size', action="store_true", default=config['model']['batch_size'])
    parser.add_argument('--n_mfcc', action="store_true", default=config['model']['n_mfcc'])
    parser.add_argument('--windows', action="store_true", default=config['model']['windows'])

    parser.add_argument('--hidden_size', action="store_true", default=config['model']['hidden_size'])
    parser.add_argument('--num_layers', action="store_true", default=config['model']['num_layers'])

    parser.add_argument('--epoch', action="store_true", default=config['model']['num_epochs'])
    parser.add_argument('--key_epoch', action="store_true", default=config['model']['key_num_epochs'])
    
    parser.add_argument('--master_key', required=False, default='./master_key.txt')
    parser.add_argument('--master_key_size', required=False, default=config['model']['master_key_size'])

    return parser.parse_args()