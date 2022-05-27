set -ex

da_replace_prob="0.4"
da_mask_prob="0"
name="220413_replace${da_replace_prob}_mask${da_mask_prob}"
data_dir="./datasets/220413/individuals"

out_dir="results/${name}"

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
--da_replace_prob ${da_replace_prob} \
--da_mask_prob ${da_mask_prob} \
"


mkdir -p $out_dir
logfile="$out_dir/train.log"

$cmd | tee $logfile
