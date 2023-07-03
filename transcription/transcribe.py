from PIL import Image, ImageFile
import torch
from pathlib import Path

from options.test_options import TestOptions
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
    opt.name = "rt7381_aligned_L100_replace0.75_hmask0.25_smask0.25"
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
    src_dir = Path("../../data/H32384_detection_result")
    dst_dir = Path("../../data/H32384_detection_result_transcribed")
    dst_dir.mkdir(exist_ok=True, parents=True)
    img_files = sorted(src_dir.glob("*.jpg"))
    print(img_files)
    for img_file in img_files:
        img = Image.open(img_file)
        print("Transcribing:", img_file.name)
        img = transcribe(img, model)
        img.save(dst_dir / img_file.name)


if __name__ == "__main__":
    main()
