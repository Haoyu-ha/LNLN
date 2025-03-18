import os
import torch
import numpy as np
import random


def save_model(save_path, epoch, model, optimizer):
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_best_results(results, best_results, epoch, model, optimizer, ckpt_root, seed, save_best_model):
    if epoch == 1:
        for key, value in results.items():
            best_results[key] = value
    else:
        for key, value in results.items():
            if (key == 'Has0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Has0_F1_score'] = results['Has0_F1_score']

                if save_best_model:
                    key_eval = 'Has0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif (key == 'Non0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Non0_F1_score'] = results['Non0_F1_score']

                if save_best_model:
                    key_eval = 'Non0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif key == 'MAE' and value < best_results[key]:
                best_results[key] = value
                # best_results['Corr'] = results['Corr']

                if save_best_model:
                    key_eval = 'MAE'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_2' and (value > best_results[key]):
                best_results[key] = value
                best_results['F1_score'] = results['F1_score']

                if save_best_model:
                    key_eval = 'Mult_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_3' or key == 'Mult_acc_5' or key == 'Mult_acc_7' or key == 'Corr':
                if value > best_results[key]:
                    best_results[key] = value

                    if save_best_model:
                        key_eval = key
                        ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
                        save_model(ckpt_path, epoch, model, optimizer)
            
            else:
                pass
    
    return best_results