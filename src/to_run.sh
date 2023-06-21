# echo Starting 0
# python train_autoregressive_predictor.py --config configs/vq/delta_v6_0.json > out_0.txt
# echo Finished 0

# USE CONCAT (Audio+Text)
# echo Starting 1
# python train_autoregressive_predictor.py -uc -ut --config configs/vq/delta_v6_1.json > out_1.txt
# echo Finished 1

# echo Starting 2
# python train_autoregressive_predictor.py -uc -ut --config configs/vq/delta_v6_2.json > out_2.txt
# echo Finished 2

# echo Starting 3
# python train_autoregressive_predictor.py -uc -ut --config configs/vq/delta_v6_3.json > out_3.txt
# echo Finished 3

# Without CONCAT
# echo Starting 4
# python train_autoregressive_predictor.py -ut --config configs/vq/delta_v6_4.json > out_4.txt
# echo Finished 4

# echo Starting 5
# python train_autoregressive_predictor.py -ut --config configs/vq/delta_v6_5.json > out_5.txt
# echo Finished 5

# echo Starting 6
# python train_autoregressive_predictor.py -ut --config configs/vq/delta_v6_6.json > out_6.txt
# echo Finished 6

# WITH CONCAT (Text+Audio)
echo Starting 7
python train_autoregressive_predictor.py -uc -ut --config configs/vq/delta_v6_7.json > out_7.txt
echo Finished 7

echo Starting 8
python train_autoregressive_predictor.py -uc -ut --config configs/vq/delta_v6_8.json > out_8.txt
echo Finished 8

echo Starting 9
python train_autoregressive_predictor.py -uc -ut --config configs/vq/delta_v6_9.json > out_9.txt
echo Finished 9


sudo shutdown now