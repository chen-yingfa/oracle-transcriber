# Oracle Transcriber

## Execution

### Prepare Data

You need to prepare data into three sub-directories: `train`, `val` and `test`, each contained image files where input A and B are concatenated horizontally. See `datasets` folder.

Conveniently, use `python3 get_oracle_data.py` to prepare to data automatically.

### Training

```bash
bash scripts/train_oracle.sh
```

### Testing

```bash
bash scripts/test_oracle.sh
```