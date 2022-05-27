from pathlib import Path
import json

from tqdm import tqdm
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from data import PairData
from model import CPair


def get_acc(labels, preds):
    assert len(labels) == len(preds)
    correct = 0
    for a, b in zip(labels, preds):
        if a == b:
            correct += 1
    return correct / len(labels)


def evaluate(model, dataset: Dataset, batch_size: int=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_logits = []
    print('*** Start Evaluate ***')
    for step, batch in enumerate(tqdm(dataloader)):
        img_a = batch['img_a'].cuda()
        img_b = batch['img_b'].cuda()
        label = batch['label'].cuda()
        logits = model(img_a, img_b)
        loss = F.cross_entropy(logits, label)
        
        # Gather result
        preds = torch.argmax(logits, dim=1).tolist()
        all_logits += logits.tolist()
        total_loss += loss.item()
        all_preds += preds
        all_labels += label.tolist()
        
    acc = get_acc(all_labels, all_preds)
        
    result = {
        'loss': total_loss / len(dataset),
        'acc': acc,
        'preds': all_preds,
        'logits': all_logits,
        'labels': all_labels,
    }
    return result

    
def train(model, data_dir: Path, output_dir: Path, num_epochs: int=10, batch_size: int=512, lr: float=1e-3):
    print('Getting data...', flush=True)
    train_data = PairData(data_dir / 'train', 'train')
    dev_data = PairData(data_dir / 'val', 'dev')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Optimizer and scheduler
    print('Preparing optimizer and scheduler...')
    num_opt_steps = len(train_dataloader)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=num_opt_steps // 2, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    print('*** Start Training ***')
    print(f'Train data: {len(train_data)}')
    print(f'Dev data: {len(dev_data)}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')
    print(f'Num epochs: {num_epochs}')
    
    total_loss = 0
    log_interval = 4
    
    for ep in range(num_epochs):
        model.train()
        print(f'*** Epoch {ep+1} ***')
        for step, batch in enumerate(train_dataloader):
            img_a = batch['img_a'].cuda()
            img_b = batch['img_b'].cuda()
            label = batch['label'].cuda()
            
            # Forward pass
            output = model(img_a, img_b)
            loss = F.cross_entropy(output, label)
            
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if step % log_interval == 0:
                print(f'Step {step / num_opt_steps:.2f}: loss = {loss.item():.6f}, lr = {scheduler.get_lr()[0]}')

        # Validation
        result = evaluate(model, dev_data, batch_size=batch_size)
        print({key: result[key] for key in ['loss', 'acc']})
        # print(result)
        
        # Save checkpoint
        ckpt_dir = output_dir / f'checkpoint-{ep}'
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        print(f'Saving to {ckpt_dir}', flush=True)
        torch.save(model.state_dict(), ckpt_dir / 'ckpt.pt')
        json.dump(result, open(ckpt_dir / 'result.json', 'w'))


def test(model, data_dir: Path, output_dir: Path, test_name: str):
    print('Loading best model by dev loss...')
    min_loss = float('inf')
    best_ckpt_dir = None
    for ckpt_dir in output_dir.glob('checkpoint-*'):
        if not ckpt_dir.is_dir(): continue
        result = json.load(open(ckpt_dir / 'result.json'))
        if result['loss'] < min_loss:
            min_loss = result['loss']
            best_ckpt_dir = ckpt_dir
    print(f'Loading from {best_ckpt_dir}')
    model.load_state_dict(torch.load(best_ckpt_dir / 'ckpt.pt'))
    
    # test_data = PairData(data_dir / 'test', 'test')
    
    # Get img paths
    gen_dir = Path(f'../oracle-transcriber/transcription/results/{test_name}/test_latest/images')
    print(f'Loading test data from {gen_dir}')
    b_paths = [img for img in gen_dir.glob('*fake_B.png')]
    a_paths = [img for img in gen_dir.glob('*real_A.png')]
    assert len(a_paths) == 147 and len(b_paths) == 147
    test_data = PairData(
        None, 'test', img_paths=(a_paths, b_paths))
    
    result = evaluate(model, test_data, batch_size=256)
    
    test_dir = output_dir / 'test' / test_name
    test_dir.mkdir(exist_ok=True, parents=True)
    json.dump(result, open(test_dir / 'result.json', 'w'))
    
    # Compute probability
    logits = result['logits']
    probs = F.softmax(torch.tensor(logits), dim=1).tolist()
    tcp = 0
    for i, label in enumerate(result['labels']):
        tcp += probs[i][label]
    tcp /= len(result['labels'])
    tcp *= 100
    open(test_dir / 'tcp.txt', 'w').write(str(tcp))
    print(f'TCP: {tcp:.2f}')


torch.manual_seed(0)
data_dir = Path('../oracle-transcriber/transcription/datasets/220413/individuals_aligned')
output_dir = Path('result/c_pair_2')

print('Getting model')
model = CPair().cuda()

# train(model, data_dir, output_dir)

# test_name = 'test'
for test_name in [
    '220413_replace0_mask1',
    '220413_replace0.4_mask0',
    '220413_replace0.8_mask0',
    '220413_aligned'
    ]:
    print(test_name)
    test(model, data_dir, output_dir, test_name=test_name)
