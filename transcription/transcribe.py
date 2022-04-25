from PIL import Image
import torch

from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform, get_params
import util


def get_opt():
    opt = TestOptions().parse(output_opt=False)
    opt.batch_size = 64
    opt.checkpoints_dir = './checkpoints'
    opt.crop_size = 128
    opt.dataset_mode = 'aligned'
    # opt.dataroot = './datasets/oracle2transcription'
    opt.dataroot = ''
    opt.direction = 'AtoB'
    opt.display_id = -1
    opt.display_winsize = 128
    opt.input_nc = 1
    opt.load_size = 128
    opt.model = 'pix2pix'
    opt.name = 'transcriber'
    opt.norm = 'batch'
    opt.netG = 'unet_128'
    opt.no_flip = True
    opt.num_threads = 0
    opt.output_nc = 1
    opt.serial_batches = True

    opt.num_threads = 0
    opt.batch_size = 1 
    opt.serial_batches = True
    opt.no_flip = True 
    opt.display_id = -1
    return opt


opt = get_opt()


def preprocess_img(img: Image) -> Image:
    def pad_and_resize(img: Image, shape: tuple=(96, 96), pad: int=255) -> Image:
        w, h = img.size
        longest = max(w, h)
        paste_pos = ((longest - w) // 2, (longest - h) // 2)
        new_img = Image.new('L', (longest, longest), color=pad)
        new_img.paste(img, paste_pos)
        new_img = new_img.resize(shape)
        return new_img
        
    img = pad_and_resize(img)
    transform_params = get_params(opt, img.size)
    transform = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1))
    img = transform(img).unsqueeze(0)
    return img


def transcribe(img: Image, model) -> Image:
    img = preprocess_img(img)
    with torch.no_grad():
        out = model.netG(img)
        out = util.util.tensor2im(out)
        img = Image.fromarray(out)
        return img


def get_model():
    # print('Creating model')
    model = create_model(opt)
    # print('Setting up model')
    model.setup(opt)
    model.eval()
    return model


def main():
    print('Getting model')
    model = get_model()

    print('Transcribing')
    # path = './datasets/oracle2transcription/test/H00072反_告.png'
    # img = Image.open(path)
    # w, h = img.size
    # img = img.crop((0, 0, w//2, h))
    # img.save('orig.png')
    
    img = Image.open('orig.png')
    img = transcribe(img, model)
    print('Saving to transcript.png')
    img.save('transcript.png')


if __name__ == '__main__':
    main()