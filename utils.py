import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
        


class AttrDict(dict):
    """ Create a dictionary that allows for attribute-style access. """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def text_input_adjust(text_input, fake_word_pos, device):
    """
    Chuẩn hóa text input và chuyển fake word positions thành fake token positions để phù hợp với tokenizer/subword của transformer.
    """
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids]) - 1
    # padding
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation (do the same as above)
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)): # broadcast each batch
        fake_token_pos = []

        # np.where return (array[...],) in this scenario
        # fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers
        fake_word_pos_decimal = torch.where(fake_word_pos[i] == 1)[0].tolist()
        # it return position indices

        subword_idx = text_input.word_ids(i) # word_ids: [None, 0, 1, 2, 2, None]
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)

        # transfer the fake word position into fake token position
        for index in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == index)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch



# LOAD CHECKPOINT
from .models.vit import interpolate_pos_embed

def load_checkpoint(args, model, optimizer, lr_scheduler):
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
        
        
from .eval.evaluate import evaluation

def evaluate(args, model, val_loader, tokenizer, optimizer, lr_scheduler, epoch, warmup_steps, device, config):
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
            'lossinfo': lossinfo
        }
    else:
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]["lr"],
            'config': config,
            'epoch': epoch,
            'lossinfo': lossinfo
        }
        
    os.makedirs('checkpoints', exist_ok=True)                        
    if (epoch % args.model_save_epoch == 0 and epoch != 0):
        torch.save(save_obj, os.path.join('checkpoints', 'checkpoint_%02d.pth' % epoch))
        
    if config['schedular']['sched'] != 'cosine_in_step':
        lr_scheduler.step(epoch + warmup_steps + 1)
        
    return val_stats