set -ex

# data_name="rubbing_531_white"
data_dir="./datasets/220413/individuals"
name="220413_aligned"
name="220413_replace0_mask1"
name="220413_replace0.4_mask0"
name="220413_replace0.8_mask0"
# out_dir="result/${name}_rubbing"
out_dir="test_result/${name}"

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
