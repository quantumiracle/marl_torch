echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`

nohup python train_pettingzoo.py --env 'pong_v1' --ram  --selfplay > log/$DATE$RAND.log &
# nohup python main.py --env 'pong_v1' --train-freq 1000 --batch-size 256 --eta 1  > log/$DATE$RAND.log &
