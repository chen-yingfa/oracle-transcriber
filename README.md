# OracleTranscriber

## 模型

是一个 41.823 M 参数的 U-Net。判别器是一个 2.765 M 参数的 CNN。

## 数据

数据处理部分不知道怎么跑起来了，但是已经处理的数据在我本地的电脑。

## 摹写

> 这是基于某个 pix2pix 仓库的，很多垃圾代码。

### 训练

直接跑

```bash
./scripts/train_oracle.sh
```

### 生成

用 `transcribe.py`，直接在 `get_opt` 函数里面手动设置参数。

主要需要关注的参数是 --checkpoint_dir 和 --name，用来指定训练好的 checkpoint，然后在 `main` 函数里面设置 `src_dir` 和 `dst_dir` 来设置输入和输出的图片。

## Developer's Note

这个是本人（陈英发）在清华大学计算机系完成的毕设的主要程序。

