import torch

from data.utils import augment_sound_data
from data.SoundDataset import SoundDataset
from omegaconf import OmegaConf
from models.crypto_model import CryptoModel


def main(cfg, stage='train') -> None:
    cfg = getattr(cfg, stage)
    if stage == 'train':
        train(cfg)
    elif stage == 'test':
        test(cfg)    
    
    return

        
def train(cfg):
    '''
    TODO: 데이터셋 제작 (O)
        - 데이터 증강 (O)
        - 데이터셋 선언 (O)
    TODO: 모델 제작 
        - LSTM Model (O)
        - Binding Model (O)
    TODO: trainer 제작

    '''
    
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CryptoModel().to(device)
    model.trainer(cfg)
    
    return

def test(cfg):
    return


if __name__ == '__main__':
    cfg = OmegaConf.load('config.yaml')
    main(cfg, stage='train')