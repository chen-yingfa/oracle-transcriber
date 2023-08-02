from PIL import Image, ImageFile
import torch
from pathlib import Path

from options.test_options import TestOptions
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch import nn
from models import create_model
from data.base_dataset import get_transform, get_params
import util


ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_opt():
    opt = TestOptions().parse(output_opt=False)
    opt.batch_size = 64
    opt.checkpoints_dir = "./checkpoints"
    opt.crop_size = 128
    opt.dataset_mode = "aligned"
    # opt.dataroot = './datasets/oracle2transcription'
    opt.dataroot = ""
    opt.direction = "AtoB"
    opt.display_id = -1
    opt.display_winsize = 128
    opt.input_nc = 1
    opt.load_size = 128
    opt.model = "pix2pix"
    opt.name = "rt7381_aligned_replace0_hmask0_smask0"
    opt.norm = "batch"
    opt.netG = "unet_128"
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


def preprocess_img(img: Image.Image) -> Image.Image:
    def pad_and_resize(
        img: Image, shape: tuple = (96, 96), pad: int = 255
    ) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        paste_pos = ((longest - w) // 2, (longest - h) // 2)
        new_img = Image.new("L", (longest, longest), color=pad)
        new_img.paste(img, paste_pos)
        new_img = new_img.resize(shape)
        return new_img

    # img = pad_and_resize(img)
    transform_params = get_params(opt, img.size)
    transform = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1))
    img = transform(img).unsqueeze(0)
    return img


def binarize(img: Image.Image, threshold: float = 0.5):
    kernel_size = 5
    
    # Convert the image to grayscale to get brightness values.
    img_gray = img.convert('L')
    pixels = img_gray.load()
    img_tensor = TF.to_tensor(img_gray)
    radius = kernel_size // 2
    conv_kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size ** 2)

    # Pad
    x = F.pad(img_tensor, [radius] * 4, mode='replicate')
    avg = F.conv2d(x, conv_kernel, stride=1)
    idx = avg > threshold
    temp = avg.flatten().tolist()
    bin_tensor = torch.ones_like(img_tensor)
    bin_tensor[avg > threshold] = 0
    bin_img = TF.to_pil_image(bin_tensor)
    return bin_img


def postprocess_img(img: Image.Image) -> Image.Image:
    return binarize(img, threshold=0.1)


def transcribe(img: Image, model) -> Image:
    img = preprocess_img(img)
    with torch.no_grad():
        out = model.netG(img)
        out = util.util.tensor2im(out)
        img = Image.fromarray(out)
        img = postprocess_img(img)
        return img


def get_model():
    print("Creating model")
    model = create_model(opt)
    print("Setting up model")
    model.setup(opt)
    model.eval()
    return model


def main():
    print("Getting model")
    model = get_model()

    print("Transcribing")
    # path = './datasets/oracle2transcription/test/H00072反_告.png'
    # img = Image.open(path)
    # w, h = img.size
    # img = img.crop((0, 0, w//2, h))
    # img.save('orig.png')
    src_dir = Path("../../H32384_case_data/cropped_boxes")
    dst_dir = Path("../H32384_transcription")
    dst_dir.mkdir(exist_ok=True, parents=True)
    img_files = sorted(src_dir.glob("*.jpg"))
    for img_file in img_files:
        img = Image.open(img_file)
        print("Transcribing:", img_file.name)
        img = transcribe(img, model)
        img.save(dst_dir / img_file.name)


if __name__ == "__main__":
    main()
