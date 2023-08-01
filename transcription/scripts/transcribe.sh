set -ex

out_dir="result/oracle_pix2pix"
da_replace_prob="0.4"
da_hmask_prob="0"
da_smask_prob="0"

cmd="python3 transcribe.py"
cmd+=" --dataroot ./datasets/oracle2transcription"
cmd+=" --checkpoints_dir ./checkpoints"
cmd+=" --name oracle_pix2pix"
cmd+=" --model pix2pix"
cmd+=" --netG unet_128"
cmd+=" --direction AtoB"
cmd+=" --dataset_mode aligned"
cmd+=" --norm batch"
cmd+=" --display_winsize 128"
cmd+=" --crop_size 128"
cmd+=" --load_size 128"
cmd+=" --input_nc 1"
cmd+=" --da_replace_prob ${da_replace_prob}"
cmd+=" --da_hmask_prob ${da_hmask_prob}"
cmd+=" --da_smask_prob ${da_smask_prob}"
cmd+=" --output_nc 1"
cmd+=" --batch_size 64"

mkdir -p $out_dir
logfile="$out_dir/test.log"

$cmd | tee $logfile
