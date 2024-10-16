# mosi
for i in $(seq 1111 1 1113)
do
  CUDA_VISIBLE_DEVICES=0 python -u train.py --config_file configs/train_mosi.yaml --seed $i >./log/mosi.txt 2>&1 &
  wait
done


# mosei
for i in $(seq 1111 1 1113)
do
  CUDA_VISIBLE_DEVICES=0 python -u train.py --config_file configs/train_mosei.yaml --seed $i >./log/mosei.txt 2>&1 &
  wait
done


# sims
for i in $(seq 1111 1 1113)
do
  CUDA_VISIBLE_DEVICES=0 python -u train.py --config_file configs/train_sims.yaml --seed $i >./log/sims.txt 2>&1 &
  wait
done