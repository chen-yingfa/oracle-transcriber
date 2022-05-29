from pathlib import Path
import matplotlib.pyplot as plt

model_name = '220413_aligned_replace0_hmask0.8_smask0'

ckpt_dir = Path('../transcription/checkpoints')
# loss_log = ckpt_dir / model_name / 'loss_log.txt'
loss_log = ckpt_dir / model_name / 'train.log'


def parse_log(loss_log):
    lines = open(loss_log, 'r').readlines()
    data = []
    for line in lines[30:]:
        if line.strip().startswith('epoch: '):
            line = line.strip().split()
            line_data = {}
            for key in ['G_GAN', 'G_L1', 'D_real', 'D_fake']:
                line_data[key] = float(line[line.index(key + ':') + 1])
            data.append(line_data)
    return data

data = parse_log(loss_log)
for key in ['G_GAN', 'G_L1', 'D_real', 'D_fake']:
    x = [i for i in range(len(data))]
    y = [data[i][key] for i in range(len(data))]
    plt.plot(x, y)
    dst_file = f'{model_name}_{key}.png'
    print(f'Saving to {dst_file}')
    plt.savefig(dst_file)
    plt.clf()