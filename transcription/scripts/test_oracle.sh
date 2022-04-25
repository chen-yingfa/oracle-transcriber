set -ex

# data_name="rubbing_531_white"
data_dir="./datasets/7104"
name="7104_noinv"
# out_dir="result/${name}_rubbing"
out_dir="result/7104"

cmd="python3 test.py \
--dataroot ${data_dir} \
--checkpoints_dir ./checkpoints
--name ${name} \
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
--batch_size 128 \
"

mkdir -p $out_dir
logfile="$out_dir/test.log"

$cmd | tee $logfile
