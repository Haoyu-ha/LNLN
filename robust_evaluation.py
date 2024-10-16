import os
import torch
import yaml
import argparse
from core.dataset import MMDataEvaluationLoader
from models.lnln import build_model
from core.metric import MetricsTop


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--key_eval', type=str, default='') 
opt = parser.parse_args()
print(opt)


def main():
    config_file = 'configs/eval_sims.yaml' if opt.config_file == '' else opt.config_file
    
    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    dataset_name = args['dataset']['datasetName']
    key_eval = args['base']['key_eval'] if opt.key_eval == '' else opt.key_eval

    model = build_model(args).to(device)
    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(dataset_name)

    missing_rate_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for cur_r in missing_rate_list:
        test_results_list = []
        if dataset_name == 'sims':
            for _, cur_seed  in enumerate([1111, 1112, 1113]):
                best_ckpt = os.path.join(f'ckpt/{dataset_name}/best_{key_eval}_{cur_seed}.pth')
                model.load_state_dict(torch.load(best_ckpt)['state_dict'])
                args['base']['missing_rate_eval_test'] = cur_r # Set missing rate

                dataLoader = MMDataEvaluationLoader(args)
        
                test_results_cur_seed = evaluate(model, dataLoader, metrics)
                # print(f'{cur_seed}: {test_results_cur_seed}')
                
                test_results_list.append(test_results_cur_seed)

            if key_eval == 'Mult_acc_2':
                Mult_acc_2_avg = (test_results_list[0]['Mult_acc_2'] + test_results_list[1]['Mult_acc_2'] + test_results_list[2]['Mult_acc_2']) / 3
                F1_score_avg = (test_results_list[0]['F1_score'] + test_results_list[1]['F1_score'] + test_results_list[2]['F1_score']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {Mult_acc_2_avg}, F1_score_avg: {F1_score_avg}')
            elif key_eval == 'Mult_acc_3':
                Mult_acc_3_avg = (test_results_list[0]['Mult_acc_3'] + test_results_list[1]['Mult_acc_3'] + test_results_list[2]['Mult_acc_3']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_3_avg: {Mult_acc_3_avg}')
            elif key_eval == 'Mult_acc_5':
                Mult_acc_5_avg = (test_results_list[0]['Mult_acc_5'] + test_results_list[1]['Mult_acc_5'] + test_results_list[2]['Mult_acc_5']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {Mult_acc_5_avg}')
            elif key_eval == 'MAE':
                MAE_avg = (test_results_list[0]['MAE'] + test_results_list[1]['MAE'] + test_results_list[2]['MAE']) / 3
                Corr_avg = (test_results_list[0]['Corr'] + test_results_list[1]['Corr'] + test_results_list[2]['Corr']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {MAE_avg}, Corr_avg: {Corr_avg}')
            
        else:
            for _, cur_seed  in enumerate([1111, 1112, 1113]):
                best_ckpt = os.path.join(f'ckpt/{dataset_name}/best_{key_eval}_{cur_seed}.pth')
                model.load_state_dict(torch.load(best_ckpt)['state_dict'])
                args['base']['missing_rate_eval_test'] = cur_r # Set missing rate

                dataLoader = MMDataEvaluationLoader(args)
        
                test_results_cur_seed = evaluate(model, dataLoader, metrics)
                
                test_results_list.append(test_results_cur_seed)

            if key_eval == 'Has0_acc_2':
                Has0_acc_2_avg = (test_results_list[0]['Has0_acc_2'] + test_results_list[1]['Has0_acc_2'] + test_results_list[2]['Has0_acc_2']) / 3
                Has0_F1_score_avg = (test_results_list[0]['Has0_F1_score'] + test_results_list[1]['Has0_F1_score'] + test_results_list[2]['Has0_F1_score']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {Has0_acc_2_avg}, F1_score_avg: {Has0_F1_score_avg}')
            elif key_eval == 'Non0_acc_2':
                Non0_acc_2_avg = (test_results_list[0]['Non0_acc_2'] + test_results_list[1]['Non0_acc_2'] + test_results_list[2]['Non0_acc_2']) / 3
                Non0_F1_score_avg = (test_results_list[0]['Non0_F1_score'] + test_results_list[1]['Non0_F1_score'] + test_results_list[2]['Non0_F1_score']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Non0_acc_2_avg: {Non0_acc_2_avg}, Non0_F1_score_avg: {Non0_F1_score_avg}')
            elif key_eval == 'Mult_acc_5':
                Mult_acc_5_avg = (test_results_list[0]['Mult_acc_5'] + test_results_list[1]['Mult_acc_5'] + test_results_list[2]['Mult_acc_5']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {Mult_acc_5_avg}')
            elif key_eval == 'Mult_acc_7':
                Mult_acc_7_avg = (test_results_list[0]['Mult_acc_7'] + test_results_list[1]['Mult_acc_7'] + test_results_list[2]['Mult_acc_7']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_7_avg: {Mult_acc_7_avg}')
            elif key_eval == 'MAE':
                MAE_avg = (test_results_list[0]['MAE'] + test_results_list[1]['MAE'] + test_results_list[2]['MAE']) / 3
                Corr_avg = (test_results_list[0]['Corr'] + test_results_list[1]['Corr'] + test_results_list[2]['Corr']) / 3
                print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {MAE_avg}, Corr_avg: {Corr_avg}')


def evaluate(model, eval_loader, metrics):
    y_pred, y_true = [], []

    model.eval()
    for cur_iter, data in enumerate(eval_loader):
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))
        sentiment_labels = data['labels']['M'].to(device)
        
        with torch.no_grad():
            out = model((None, None, None), incomplete_input)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(sentiment_labels.cpu())
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    return results



if __name__ == '__main__':
    main()
