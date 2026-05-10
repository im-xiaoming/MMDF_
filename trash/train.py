import argparse
import ruamel.yaml as yaml
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from tqdm import tqdm


from models.HAMMER import HAMMER
from utils import text_input_adjust
from eval.evaluate import evaluation


def train(args, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    print_freq = 100   
    step_size = 100
    warmup_iterations = warmup_steps * step_size  

    global_step = epoch * len(data_loader)
    avg_loss = []

    # text = caption; fake_word_pos = fake_text_pos_list
    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(data_loader):

        if config['schedular']['sched'] == 'cosine_in_step':
            scheduler.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args, config)        

        # zero gradient
        optimizer.zero_grad()
  
        # move to device
        image = image.to(device, non_blocking=True) 
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        
        # forward
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device) # text input is list token
 
        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader)) 
        
        loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC = model(image, label, text_input, fake_image_box, fake_token_pos, alpha=alpha)  
            
        loss = config['loss_MAC_wgt'] * loss_MAC \
             + config['loss_BIC_wgt'] * loss_BIC \
             + config['loss_bbox_wgt'] * loss_bbox \
             + config['loss_giou_wgt'] * loss_giou \
             + config['loss_TMG_wgt'] * loss_TMG \
             + config['loss_MLC_wgt'] * loss_MLC \
        
        # backward
        loss.backward()
        optimizer.step()    
        
        print(f"loss_MAC: {loss_MAC.item()}")
        print(f"loss_BIC: {loss_BIC.item()}")
        print(f"loss_bbox: {loss_bbox.item()}")
        print(f"loss_giou: {loss_giou.item()}")
        print(f"loss_TMG: {loss_TMG.item()}")
        print(f"loss_MLC: {loss_MLC.item()}")
        print(f"loss: {loss.item()}")
        print(f"lr: {optimizer.param_groups[0]['lr']}")

        avg_loss.append(loss.item())
        
        if epoch == 0 and i % step_size==0 and i <= warmup_iterations and config['schedular']['sched'] != 'cosine_in_step': 
            scheduler.step(i // step_size)   

        global_step += 1
         
    return np.mean(avg_loss)    



    
def main_worker(gpu, args, config):

    ###########################################################
    device = torch.device('cuda')
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']  # 50
    warmup_steps = config['schedular']['warmup_epochs']  # 10
    best = 0
    best_epoch = 0  

    #### Dataset #### 
    train_dataset, val_dataset = create_dataset(config)
    

    train_loader, val_loader = create_loader([train_dataset, val_dataset],
                                batch_size=[config['batch_size_train']] + [config['batch_size_val']], 
                                num_workers=[4, 4], 
                                is_trains=[True, False]
    )

    # LOAD BERT TOKENIZER
    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)

    #### Model #### 
    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)   
    
    # optimizer
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model) # optim/optim_factory.py
    
    # scheduler
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    if config['schedular']['sched'] == 'cosine_in_step':
        args.lr = config['optimizer']['lr']
    
    
    # CHECKPOINT
    if args.checkpoint:   
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)   
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
                 
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    ################################## TRAINING ############################################
    for epoch in range(start_epoch, max_epoch):
        
        # train
        train_stats = train(args, model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)
        
        # evaluation 
        AUC_cls, ACC_cls, EER_cls, \
        MAP, OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok \
        = evaluation(args, model, val_loader, tokenizer, device, config)

        lossinfo = {
            'AUC_cls': round(AUC_cls*100, 4),                                                                                                  
            'ACC_cls': round(ACC_cls*100, 4),                                                                                                  
            'EER_cls': round(EER_cls*100, 4),                                                                                                  
            'MAP': round(MAP*100, 4),                                                                                                  
            'OP': round(OP*100, 4),                                                                                                  
            'OR': round(OR*100, 4), 
            'OF1': round(OF1*100, 4), 
            'CP': round(CP*100, 4), 
            'CR': round(CR*100, 4), 
            'CF1': round(CF1*100, 4), 
            'OP_k': round(OP_k*100, 4), 
            'OR_k': round(OR_k*100, 4), 
            'OF1_k': round(OF1_k*100, 4), 
            'CP_k': round(CP_k*100, 4), 
            'CR_k': round(CR_k*100, 4), 
            'CF1_k': round(CF1_k*100, 4), 
            'IOU_score': round(IOU_score*100, 4),                                                                                                  
            'IOU_ACC_50': round(IOU_ACC_50*100, 4),                                                                                                  
            'IOU_ACC_75': round(IOU_ACC_75*100, 4),                                                                                                  
            'IOU_ACC_95': round(IOU_ACC_95*100, 4),                                                                                                  
            'ACC_tok': round(ACC_tok*100, 4),                                                                                                  
            'Precision_tok': round(Precision_tok*100, 4),                                                                                                  
            'Recall_tok': round(Recall_tok*100, 4),                                                                                                  
            'F1_tok': round(F1_tok*100, 4),                                                                                                  
                } 

        #============ evaluation info ============#
        val_stats = {"AUC_cls": "{:.4f}".format(AUC_cls*100),
                     "ACC_cls": "{:.4f}".format(ACC_cls*100),
                     "EER_cls": "{:.4f}".format(EER_cls*100),
                     "MAP": "{:.4f}".format(MAP*100),
                     "OP": "{:.4f}".format(OP*100),
                     "OR": "{:.4f}".format(OR*100),
                     "OF1": "{:.4f}".format(OF1*100),
                     "CP": "{:.4f}".format(CP*100),
                     "CR": "{:.4f}".format(CR*100),
                     "CF1": "{:.4f}".format(CF1*100),
                     "OP_k": "{:.4f}".format(OP_k*100),
                     "OR_k": "{:.4f}".format(OR_k*100),
                     "OF1_k": "{:.4f}".format(OF1_k*100),
                     "CP_k": "{:.4f}".format(CP_k*100),
                     "CR_k": "{:.4f}".format(CR_k*100),
                     "CF1_k": "{:.4f}".format(CF1_k*100),
                     "IOU_score": "{:.4f}".format(IOU_score*100),
                     "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50*100),
                     "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75*100),
                     "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95*100),
                     "ACC_tok": "{:.4f}".format(ACC_tok*100),
                     "Precision_tok": "{:.4f}".format(Precision_tok*100),
                     "Recall_tok": "{:.4f}".format(Recall_tok*100),
                     "F1_tok": "{:.4f}".format(F1_tok*100),
        }
        
        if config['schedular']['sched'] != 'cosine_in_step':
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
        else:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': optimizer.param_groups[0]["lr"],
                'config': config,
                'epoch': epoch,
            }
                               
        if (epoch % args.model_save_epoch == 0 and epoch!=0):
            torch.save(save_obj, 'checkpoint_%02d.pth' % epoch) 
        if float(val_stats['AUC_cls']) > best:
            torch.save(save_obj, 'checkpoint_best.pth') 
            best = float(val_stats['AUC_cls'])
            best_epoch = epoch 

        if config['schedular']['sched'] != 'cosine_in_step':
            lr_scheduler.step(epoch + warmup_steps + 1)  
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['pytorch', 'slurm', 'mpi'], default='pytorch',
                        help='job launcher')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=20)
    parser.add_argument('--token_momentum', default=False, action='store_true') # if specified: True

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader) # config = train.yaml

    # main(args, config)
    # THIS ALWAYS PYTORCH
    main_worker(gpu=0, args=args, config=config)