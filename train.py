import os
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results
from models.lnln import build_model
from core.metric import MetricsTop 


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--seed', type=int, default=-1) 
opt = parser.parse_args()
print(opt)

def main():
    best_valid_results, best_test_results = {}, {}

    config_file = 'configs/train_sims.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("ckpt root :", ckpt_root)

    model = build_model(args).to(device)

    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args['base']['lr'],
                                 weight_decay=args['base']['weight_decay'])
    scheduler_warmup = get_scheduler(optimizer, args)

    loss_fn = MultimodalLoss(args)

    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])


    for epoch in range(1, args['base']['n_epochs']+1):
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)

        if args['base']['do_validation']:
            valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=False)
            print(f'Current Best Valid Results: {best_valid_results}')

        test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=True)
        print(f'Current Best Test Results: {best_test_results}\n')

        scheduler_warmup.step()

        # if epoch == 1:
        #     for key, value in test_results.items():
        #         best_test_results[key] = value
        # else:
        #     for key, value in test_results.items():
        #         if (key == 'Has0_F1_score') and (value + test_results['Has0_acc_2'] > best_test_results[key] + best_test_results['Has0_acc_2']):
        #             best_test_results[key] = value
        #             best_test_results['Has0_acc_2'] = test_results['Has0_acc_2']

        #             key_eval = 'Has0_acc_2'
        #             ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
        #             save_model(ckpt_path, epoch, model, optimizer)
                
        #         elif (key =='Non0_F1_score') and (value + test_results['Non0_acc_2'] > best_test_results[key] + best_test_results['Non0_acc_2']):
        #             best_test_results[key] = value
        #             best_test_results['Non0_acc_2'] = test_results['Non0_acc_2']

        #             key_eval = 'Non0_acc_2'
        #             ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
        #             save_model(ckpt_path, epoch, model, optimizer)
                
        #         elif key == 'MAE' and value > best_test_results[key]:
        #             best_test_results[key] = value
        #             best_test_results['Corr'] = test_results['Corr']

        #             key_eval = 'MAE'
        #             ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
        #             save_model(ckpt_path, epoch, model, optimizer)

        #         elif key == 'F1_score' and (value + test_results['Mult_acc_2'] > best_test_results[key] + best_test_results['Mult_acc_2']):
        #             best_test_results[key] = value
        #             best_test_results['Mult_acc_2'] = test_results['Mult_acc_2']

        #             key_eval = 'Mult_acc_2'
        #             ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
        #             save_model(ckpt_path, epoch, model, optimizer)

        #         elif key == 'Mult_acc_3' or key == 'Mult_acc_5' or key == 'Mult_acc_7':
        #             best_test_results[key] = value

        #             key_eval = key
        #             ckpt_path = os.path.join(ckpt_root, f'best_{key_eval}_{seed}.pth')
        #             save_model(ckpt_path, epoch, model, optimizer)
                
        #         else:
        #             assert False, f'Unknown metrics: {key}'
        
        # print(f'Current Best Test Results: {best_test_results}\n')



def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)
        effectiveness_labels = torch.cat([torch.ones(len(sentiment_labels)*8), torch.zeros(len(sentiment_labels)*8)]).long().to(device)
        label = {'sentiment_labels': sentiment_labels, 'completeness_labels': completeness_labels, 'effectiveness_labels': effectiveness_labels}

        out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                loss_dict[key] = value.item()
            # loss_dict['loss'] = loss['loss'].item()
            # loss_dict['l_cc'] = loss['l_cc'].item()
            # loss_dict['l_adv'] = loss['l_adv'].item()
            # loss_dict['l_rec'] = loss['l_rec'].item()
            # loss_dict['l_sp'] = loss['l_sp'].item()
        else:
            for key, value in loss.items():
                loss_dict[key] += value.item()
            # loss_dict['loss'] += loss['loss'].item()
            # loss_dict['l_cc'] += loss['l_cc'].item()
            # loss_dict['l_adv'] += loss['l_adv'].item()
            # loss_dict['l_rec'] += loss['l_rec'].item()
            # loss_dict['l_sp'] += loss['l_sp'].item()

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    loss_dict = {key: value / (cur_iter+1) for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')




def evaluate(model, eval_loader, loss_fn, epoch, metrics):
    loss_dict = {}

    y_pred, y_true = [], []

    model.eval()
    
    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)
        effectiveness_labels = torch.cat([torch.ones(len(sentiment_labels)*8), torch.zeros(len(sentiment_labels)*8)]).long().to(device)
        label = {'sentiment_labels': sentiment_labels, 'completeness_labels': completeness_labels, 'effectiveness_labels': effectiveness_labels}
        
        with torch.no_grad():
            out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item()
                except:
                    loss_dict[key] += value
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    
    # print(f'Test Loss Epoch {epoch}: {loss_dict}')
    # print(f'Test Results Epoch {epoch}: {results}')

    return results


if __name__ == '__main__':
    main()


