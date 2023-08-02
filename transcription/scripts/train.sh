set -ex

da_replace_prob="0"
da_hmask_prob="0"
da_smask_prob="0"
name="rt7381_aligned_replace${da_replace_prob}_hmask${da_hmask_prob}_smask${da_smask_prob}"
data_dir="./datasets/rt7381/individuals_96_aligned"

out_dir="checkpoints/${name}"

pyargs=" -W ignore "

cmd="python3 ${pyargs} train.py"
cmd+=" --dataroot ${data_dir}"
cmd+=" --checkpoints_dir ./checkpoints"
cmd+=" --name ${name}"
cmd+=" --model pix2pix"
cmd+=" --netG unet_128"
cmd+=" --direction AtoB"
cmd+=" --lambda_L1 100"
cmd+=" --dataset_mode aligned"
cmd+=" --norm batch"
cmd+=" --pool_size 0"
cmd+=" --display_winsize 128"
cmd+=" --crop_size 128"
cmd+=" --load_size 128"
cmd+=" --input_nc 1"
cmd+=" --output_nc 1"
cmd+=" --batch_size 128"
cmd+=" --n_epochs 100"
cmd+=" --n_epochs_decay 100"
cmd+=" --save_epoch_freq 20"
cmd+=" --no_dropout"
cmd+=" --da_replace_prob ${da_replace_prob}"
cmd+=" --da_hmask_prob ${da_hmask_prob}"
cmd+=" --da_smask_prob ${da_smask_prob}"
cmd+=" --print_freq 64"
cmd+=" --gpu_ids 0"

mkdir -p $out_dir
logfile="$out_dir/train.log"

$cmd | tee $logfile
