set -ex

out_dir="result/oracle_pix2pix"

cmd="python3 transcribe.py \
--dataroot ./datasets/oracle2transcription \
--checkpoints_dir ./checkpoints
--name oracle_pix2pix \
--model pix2pix \
--netG unet_128 \
--direction AtoB \
--dataset_mode aligned \
--norm batch \
--display_winsize 128 \
--crop_size 128 \
--load_size 128 \
--input_nc 1 \
--output_nc 1 \
--batch_size 64 \
"

mkdir -p $out_dir
logfile="$out_dir/test.log"

$cmd | tee $logfile
