set -ex

data_name="7104"
name="${data_name}_noinv"
data_dir="./datasets/${data_name}"
out_dir="result/temp_${name}"

pyargs=" -W ignore "

cmd="python3 ${pyargs} train.py \
--dataroot ${data_dir} \
--checkpoints_dir ./checkpoints \
--name ${name} \
--model pix2pix \
--netG unet_128 \
--direction AtoB \
--lambda_L1 100 \
--dataset_mode aligned \
--norm batch \
--pool_size 0 \
--display_winsize 128 \
--crop_size 128 \
--load_size 128 \
--input_nc 1 \
--output_nc 1 \
--batch_size 128 \
--n_epochs 160 \
--n_epochs_decay 160 \
--save_epoch_freq 40 \
"


mkdir -p $out_dir
logfile="$out_dir/train.log"

$cmd | tee $logfile
