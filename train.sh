echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# nohup python train_pettingzoo.py --env 'pong_v1' --ram  --selfplay > log/$DATE$RAND.log &
# nohup python train_pettingzoo_mp_vecenv.py --env 'pong_v1' --ram --num-envs 3 --selfplay > log/$DATE$RAND.log &
#nohup python train_pettingzoo_mp_vecenv.py --env 'pong_v1' --num-envs 3 --selfplay > log/$DATE$RAND.log &
#nohup python main.py --env 'pong_v1' --train-freq 1000 --batch-size 256 --eta 1  > log/$DATE$RAND.log &
nohup python train_pettingzoo_mp_vecenv.py --env slimevolley_v0 --ram --num-envs 3 --selfplay > log/$DATE$RAND.log &

