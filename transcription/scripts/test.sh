# data_name="rubbing_531_white"
data_dir="./datasets/rt7381/individuals_aligned"
name="220413_aligned"
name="220413_replace0_mask1"
name="220413_replace0.4_mask0"
name="220413_replace0.8_mask0"
name="220413"
name="220413_aligned_replace0_hmask0.8_smask0.8"
name="220413_aligned_replace0_hmask0.8_smask0.4"
name="rt7381_aligned_replace0_hmask0.8_smask0"
name="rt7381_aligned_replace0_hmask0.8_smask0.8"
# name="rt7381_aligned_replace0_hmask0.5_smask0"
name="rt7381_aligned_replace0.4_hmask0_smask0"
# name="rt7381_aligned_replace0_hmask0.8_smask0"
# out_dir="result/${name}_rubbing"
out_dir="test_result/${name}"

cmd="python3 test.py"
cmd+=" --dataroot ${data_dir}"
cmd+=" --checkpoints_dir ./checkpoints"
cmd+=" --name ${name}"
cmd+=" --model pix2pix"
cmd+=" --netG unet_128"
cmd+=" --direction AtoB"
cmd+=" --dataset_mode aligned"
cmd+=" --norm batch"
cmd+=" --display_winsize 128"
cmd+=" --crop_size 128"
cmd+=" --load_size 128"
cmd+=" --input_nc 1"
cmd+=" --output_nc 1"
cmd+=" --batch_size 128"
cmd+=" --no_dropout"

mkdir -p $out_dir
logfile="$out_dir/test.log"

$cmd | tee $logfile