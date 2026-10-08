# 计算机视觉实验

本仓库记录计算机视觉实验内容。

当前版本：`0.1.0`。版本记录见 [CHANGELOG.md](CHANGELOG.md)。

## 实验一：开发库环境验证

检查 PyTorch、OpenCV 和 Scikit-learn 的导入及基础功能。以下是实验文档中结果的可视化摘要：

![PyTorch 验证结果截图](exp1/assets/01_pytorch_verification.png)

![OpenCV 验证结果截图](exp1/assets/02_opencv_verification.png)

![Scikit-learn 验证结果截图](exp1/assets/03_sklearn_verification.png)

![综合验证结果截图](exp1/assets/04_all_packages_verification.png)

详细实验过程与分析见 [实验一说明](exp1/readme.md)。验证结果以 PNG 图片形式展示。

## 实验二：图像增强

使用提供的图片完成 BGR/RGB 与灰度转换、椒盐和高斯噪声添加、均值/中值/高斯滤波，以及手动彩色中值滤波。所有计算均使用原图 `1080×1578` 分辨率，固定随机种子便于复现。

![两类噪声与三种滤波的结果对比](exp2/result/filter_results_2x3.png)

在本次图片和参数下，中值滤波对椒盐噪声效果最好；均值滤波在高斯噪声组的 PSNR 与 SSIM 均最高。手动中值与 OpenCV 输出逐通道完全相同。

- [实验二报告](exp2/readme.md)
- [实验 Notebook](exp2/code.ipynb)
- [可运行源码](exp2/experiment.py)
- [实验指标](exp2/result/metrics.csv)

在仓库根目录运行（Python 3.12，Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r exp2/requirements.txt
.\.venv\Scripts\python.exe exp2/experiment.py
```
