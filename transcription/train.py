"""
General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
"""
import time
from options.train_options import TrainOptions
from data import create_dataloader
from models import create_model
# from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataloader(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    global_step = 0                # the total number of training iterations

    print('*** Start Training ***')
    print(f'batch_size: {opt.batch_size}')
    print(f'epochs: {opt.n_epochs}')
    print(f'The number of training images: {dataset_size}')

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for ep in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1): 
        print(f'*** Start training epoch {ep} ***', flush=True)
        
        epoch_start_time = time.time()  # timer for entire epoch
        # iter_data_time = time.time()    # timer for data loading per iteration
        ep_step = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for step, batch in enumerate(dataset):  # inner loop within one epoch
            # iter_start_time = time.time()  # timer for computation per iteration
            # if global_step % opt.print_freq == 0:
                # t_data = iter_start_time - iter_data_time

            global_step += 1
            ep_step += 1
            model.set_input(batch)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # # display images on visdom and save images to a HTML file
            # if global_step % opt.display_freq == 0:
            #     save_result = global_step % opt.update_html_freq == 0
            #     model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), ep, save_result)

            # print training losses and save logging information to the disk
            if global_step % opt.print_freq == 0:
                losses = model.get_current_losses()
                # t_comp = (time.time() - iter_start_time) / opt.batch_size
                # visualizer.print_current_losses(ep, ep_step, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(ep, float(ep_step) / dataset_size, losses)
                print(f'epoch: {ep} '
                      f'step: {ep_step} '
                      f'time: {time.time() - epoch_start_time:.2f} '
                      f'G_GAN: {losses["G_GAN"]} '
                      f'G_L1: {losses["G_L1"]} '
                      f'D_real: {losses["D_real"]} '
                      f'D_fake: {losses["D_fake"]} ',
                      flush=True)

            # cache our latest model every <save_latest_freq> iterations
            if global_step % opt.save_latest_freq == 0:
                print(f'saving the latest model (epoch {ep}, global_step {global_step})', flush=True)
                save_suffix = 'iter_%d' % global_step if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if ep % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print(f'saving the model at the end of epoch {ep}, iters {global_step}', flush=True)
            model.compute_visuals()
            model.save_networks('latest')
            model.save_networks(ep)

        time_elapsed_epoch = time.time() - epoch_start_time
        print(f'End of epoch {ep}/{opt.n_epochs} \t Time Taken: {time_elapsed_epoch} sec')
