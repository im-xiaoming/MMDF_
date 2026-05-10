import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.dataset import DGM4_Dataset
from dataset.randaugment import RandomAugment

def create_dataset(config):
    
    # TRANSFORM
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    train_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])    
    
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    
    # CREATE DATASET
    train_dataset = DGM4_Dataset(config=config, ann_file=config['train_file'], transform=train_transform, max_words=config['max_words'], is_train=True)              
    val_dataset = DGM4_Dataset(config=config, ann_file=config['val_file'], transform=test_transform, max_words=config['max_words'], is_train=False)              
    return train_dataset, val_dataset   


def create_loader(datasets, batch_size, num_workers, is_trains):
    loaders = []
    for dataset, bs, n_worker, is_train in zip(datasets, batch_size, num_workers, is_trains):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    