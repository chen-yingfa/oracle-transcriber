# Data preprocessing steps

预处理模块主要负责对应汉达文库的数据和OCR获取的摹本数据，并对图片进行预处理，整理成适合进行 Image to Image translation 的格式。有两种方式：

1. 自己求对应关系。
2. 使用宋晨阳整理好的对应关系。

## 宋晨阳的对应数据

1. 对齐噪声字和摹写字（`gen_pairs.py`）
2. 统一图片大小（`resize.py`）

## 自己求对应关系

1. 整理摹写数据（`ocr_data.py`）
2. 对齐噪声字和摹写字（`gen_pairs.py`）
3. 统一图片大小（`resize.py`）

## 详细步骤

### Step 1：整理摹写数据

执行：`python3 ocr_data.py`

输出文件夹：`../data/{name}/ocr`

从 OCR 结果中拷贝过来。各图片划分在以所属甲片的编号（如 00035）命名的文件夹。即，`../data/ocr/00035` 含有属于 H00035 的甲骨字图片。

忽略掉所有无效的甲片编号或者无效汉字标签的文件。

### Step 2：对齐噪声字和摹写字

执行：`python3 gen_pairs.py` 

输出文件夹：`./data/{name}/pairs`

对于每个图片，根据甲片编号找到甲片，若能唯一对应到一个甲骨字，则将摹写图片和对应的原始甲骨字图片进行**对齐**，将对齐后的一对图片左右拼接。

### Step 3：统一图片大小

执行：`python3 resize.py`

输出文件夹：`./data/{name}/oracle2transcription`

将对齐完的图片进行大小调整，默认大小为 (96, 96)，一对就是 (192, 96)

## 摹写对齐

为了让模型更好的学习的摹写规律，我们希望输出数据中，摹写的笔画和带有噪声的甲骨字的刻痕尽量重合。对所谓“对齐”就是求这个重合的位置。

### 摹写数据的预处理

1. 转成黑底白字
2. 二值化（0或255）
3. 裁剪至bounding box（即去掉多余的黑色）

### 对齐

遍历摹写图片的不同大小，不同位置，求使图片距离最小的位置。

距离函数：$ Distance(A,B) = Avg((A-B)^2) $