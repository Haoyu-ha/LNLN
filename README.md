# Towards Robust Multimodal Sentiment Analysis with Incomplete Data

Pytorch implementation of the paper: 
> **[Towards Robust Multimodal Sentiment Analysis with Incomplete Data](https://openreview.net/pdf?id=mYEjc7qGRA)**

> This is a reorganized code, if you find any bugs please contact me. Thanks.

## Content
- [Data Preparation](#Data-preparation)
- [Environment](#Environment)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Note](#Note)
- [Corrigendum](#Corrigendum)
- [Citation](#Citation)

## Data Preparation
MOSI/MOSEI/CH-SIMS Download: Please see [MMSA](https://github.com/thuiar/MMSA)

## Environment
The basic training environment for the results in the paper is Pytorch 2.2.1, Python 3.11.7 with NVIDIA Tesla A40. 

## Training
You can quickly run the code with the following command:
```
bash train.sh
```

## Evaluation
After the training is completed, the checkpoints corresponding to the three random seeds (1111,1112,1113) can be used for evaluation. For example, evaluate the the model's binary classification accuracy in MOSI:
```
CUDA_VISIBLE_DEVICES=0 python robust_evaluation.py --config_file configs/eval_mosi.yaml --key_eval Has0_acc_2
```

## Note
1. This work builds upon our previous works [ALMT](https://github.com/Haoyu-ha/ALMT), which was published in EMNLP 2023.
2. Due to the regression metrics (such as MAE and Corr) and classification metrics (such as acc2 and F1) focus on different aspects of model performance. A model that achieves the lowest error in sentiment intensity prediction does not necessarily perform best in classification tasks. To comprehensively demonstrate the capabilities of the models, all the results of all models in the comparisons are selected as the best-performing checkpoint for each type of metric. This means that the classification metrics (such as acc2 and F1) and regression metrics (such as MAE and Corr) correspond to different epochs of the same training process. If you wish to compare the performance of models across different metrics at the same epoch, we recommend you rerun this code.


## Corrigendum
1. In **Table 9**, the **Acc-5** of the CENET at the r=0.7 is incorrectly reported as `59.86%`. The correct value should be **23.57%**. This error impacts the overall robustness evaluation in **Table 2**, where the Acc-5 of CENET is revised from `37.25%` to **33.62%**. The mistake occurred during manual filling in the values for multiple tables. This correction does not alter the performance of proposed LNLN, nor does it affect the original analysis and conclusions of the paper. We sincerely apologize for the oversight and thank the **readers** for identifying this issue.


## Citation

- [Towards Robust Multimodal Sentiment Analysis with Incomplete Data](https://arxiv.org/abs/2409.20012)

Please cite our paper if you find our work useful for your research:

```
@inproceedings{zhang-etal-2024-lnln,
    title = "Towards Robust Multimodal Sentiment Analysis with Incomplete Data",
    author = "Zhang, Haoyu and 
              Wang, Wenbin and 
              Yu, Tianshu",
    booktitle = "The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)",
    year = "2024"
}
```
