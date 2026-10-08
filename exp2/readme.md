# 实验二：图像基本操作与滤波

实验版本：`0.1.0`　实验日期：`2026-10-08`

实验内容参考 [CV_exp / 2nd.exp](https://github.com/Guyudong1/CV_exp/tree/main/2nd.exp)。本次使用用户提供的图片 [input.jpg](assets/input.jpg)，以原始分辨率 **1080×1578** 完成全部计算，不缩放输入图像。实现代码见 [experiment.py](experiment.py)，交互式实验见 [code.ipynb](code.ipynb)。

## 实验目的：

<p style="text-indent: 2em;">掌握 OpenCV 图像读取、像素访问、颜色通道转换与灰度化方法；使用 Scikit-image 为彩色图像添加椒盐噪声和高斯噪声；比较均值、中值和高斯滤波的去噪效果；在不调用现成滤波函数的条件下实现彩色图像中值滤波，并验证手动实现与 OpenCV 结果的一致性。</p>

## 实验内容：

### 1.读取图像、访问像素并转换颜色通道：

<p style="text-indent: 2em;">通过 OpenCV 读取用户图片。为支持中文路径，先读取文件字节，再使用 <code>cv2.imdecode</code> 解码。OpenCV 默认使用 BGR 通道顺序，Matplotlib 的彩色图像显示则按 RGB 解释，因此显示前需要执行 BGR→RGB 转换。</p>

```python
bgr = cv2.imdecode(
    np.frombuffer(input_path.read_bytes(), dtype=np.uint8),
    cv2.IMREAD_COLOR,
)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

print(bgr.shape)       # (1578, 1080, 3)
print(bgr[100, 100])   # [111 132 123]
print(rgb[100, 100])   # [123 132 111]
```

数组坐标 `[100, 100]` 表示从零开始计数的第 100 行、第 100 列；彩色图像的数据类型为 `uint8`，每个通道取值范围为 `0～255`。BGR→RGB 交换第一个和第三个通道，以匹配显示接口的解释顺序，**不会改变图像真实颜色**。直接把 BGR 数组交给 `imshow` 才会造成红蓝显示错位。

### 2.转换并保存灰度图像：

<p style="text-indent: 2em;">使用 <code>cv2.COLOR_BGR2GRAY</code> 将三个颜色通道转换为单通道亮度表示，并用灰度色图显示。灰度化依据不同颜色通道的加权贡献，而非简单求平均。</p>

```python
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
write_image(output_dir / "gray_image.png", gray)

plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
```

灰度图形状为 `(1578, 1080)`，像素 `[100, 100]` 的灰度值为 `127`。下图从左到右分别为误按 RGB 显示的 BGR 数组、正确显示的 RGB 图像和灰度图像。

![BGR 显示、RGB 显示与灰度图对比](result/color_spaces.png)

完整灰度输出见 [gray_image.png](result/gray_image.png)。

### 3.添加椒盐噪声和高斯噪声：

<p style="text-indent: 2em;">沿用参考实验的噪声参数：椒盐噪声 <code>amount=0.4</code>，盐与椒的比例各为 0.5；高斯噪声均值为 <code>0.2</code>、方差为 <code>0.03</code>。两类噪声分别使用固定随机种子 2026 和 2027，使本次结果可以复现。</p>

```python
sp_float = random_noise(
    rgb, mode="s&p", amount=0.4,
    rng=np.random.default_rng(2026),
)
gaussian_float = random_noise(
    rgb, mode="gaussian", mean=0.2, var=0.03,
    rng=np.random.default_rng(2027),
)

def to_uint8(image):
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

sp_noisy = to_uint8(sp_float)
gaussian_noisy = to_uint8(gaussian_float)
```

`random_noise` 在归一化到 `[0, 1]` 的浮点图像上加噪。输出统一裁剪、四舍五入并转回 `uint8`，之后各滤波方法使用同一份噪声输入。

`amount=0.4` 对多通道图像表示约 40% 的**通道元素**被抽中处理，不能解释为恰好 40% 的完整像素被污染。在三个通道独立采样时，至少一个通道被抽中的理论像素比例为 `1−(1−0.4)³=78.4%`。本次实际改变的通道元素比例为 **38.928%**，至少一个通道发生改变的像素比例为 **76.913%**。原图中已有的 `0` 或 `255` 通道值可能被替换为相同值，因此“抽中处理”不一定产生可观测的数值变化。

高斯噪声的 `mean=0.2` 为正值，添加前的名义平均偏移相当于 `51` 个通道灰度级；输出超过 `[0, 1]` 的值会被裁剪。本次达到上界 `1` 的通道元素占 **33.235%**，达到下界 `0` 的占 **0.014%**，转换为 `uint8` 后相对原图的平均通道偏差为 **+36.374**。因此，这组高斯噪声会使图像整体偏亮，并使部分高亮细节饱和。

![原图、椒盐噪声与高斯噪声对比](result/noise_comparison.png)

### 4.比较三种 OpenCV 滤波方法：

<p style="text-indent: 2em;">分别对两类噪声图像应用均值滤波、中值滤波和高斯滤波。统一使用 5×5 窗口和复制边界，高斯滤波的 <code>sigmaX=0</code> 表示由 OpenCV 根据窗口大小确定标准差。将两类噪声的三种结果排列为 2×3 子图，以原始 RGB 图像作为指标参考。</p>

```python
kernel_size = 5
mean_image = cv2.blur(
    noisy, (kernel_size, kernel_size),
    borderType=cv2.BORDER_REPLICATE,
)
median_image = cv2.medianBlur(noisy, kernel_size)
gaussian_image = cv2.GaussianBlur(
    noisy, (kernel_size, kernel_size), 0,
    borderType=cv2.BORDER_REPLICATE,
)
```

均值滤波对邻域等权平均；中值滤波逐通道选择邻域的中位数；高斯滤波对邻域进行高斯权重平均。中值滤波天然使用复制边界，另外两种方法显式指定相同边界方式，避免边界条件差异影响比较。

![两类噪声下三种滤波结果的 2×3 对比](result/filter_results_2x3.png)

图中第一行为椒盐噪声、第二行为高斯噪声；每行从左到右依次为均值滤波、中值滤波、高斯滤波。

### 5.手动实现中值滤波并保存结果：

<p style="text-indent: 2em;">将椒盐噪声图像拆分为三个通道，对边界进行复制填充，再取每个 5×5 邻域的中位数。代码使用 NumPy 组织邻域并求中位数，不调用 OpenCV、Scikit-image 或其他库的现成滤波函数。按 64 行分块处理，以控制原始分辨率图像的中间内存占用。</p>

```python
def manual_median_filter_color(image, kernel_size=5, block_rows=64):
    height = image.shape[0]
    pad = kernel_size // 2
    result = np.empty_like(image)
    for channel_index in range(3):
        channel = np.pad(image[:, :, channel_index], pad, mode="edge")
        for start in range(0, height, block_rows):
            end = min(start + block_rows, height)
            windows = sliding_window_view(
                channel[start:end + 2 * pad, :],
                (kernel_size, kernel_size),
            )
            result[start:end, :, channel_index] = np.median(
                windows, axis=(-2, -1),
            )
    return result

manual = manual_median_filter_color(sp_noisy)
write_image(output_dir / "denoised_sp.png", manual)
```

每个通道的 5×5 邻域含 25 个元素，排序后的第 13 个值就是中位数。分块求值与逐像素计算的数学过程相同，图像边界的 `mode="edge"` 与 OpenCV 的 `BORDER_REPLICATE` 一致。脚本中的完整函数还检查输入类型、窗口大小及分块参数。

![椒盐输入、OpenCV 中值滤波与手动中值滤波对比](result/manual_median.png)

手动结果与 OpenCV 中值滤波在整幅 `1080×1578×3` 图像上逐元素一致：最大绝对差为 **0**，不同通道值数量为 **0**。保存的完整去噪结果见 [denoised_sp.png](result/denoised_sp.png)。

## 实验结果与分析：

<p style="text-indent: 2em;">下表来自本次实际运行生成的指标文件。MSE 越小表示像素误差越小，PSNR 和 SSIM 越大表示与原图越接近；平均误差为输出减去原图后在全部通道上取平均，正值表示平均通道值偏高。所有指标均在完整原图尺寸下计算，PSNR 和 SSIM 的数据范围统一设为 255。</p>

| 噪声类型 | 处理方法 | MSE | PSNR / dB | SSIM | 平均误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 椒盐噪声 | 加噪后，未滤波 | 8404.61 | 8.886 | 0.0185 | −17.913 |
| 椒盐噪声 | 均值滤波 | 1094.99 | 17.737 | 0.2704 | −17.914 |
| 椒盐噪声 | 中值滤波 | **49.09** | **31.221** | **0.8819** | +0.099 |
| 椒盐噪声 | 高斯滤波 | 1351.56 | 16.822 | 0.1713 | −17.916 |
| 高斯噪声 | 加噪后，未滤波 | 2773.50 | 13.701 | 0.1528 | +36.374 |
| 高斯噪声 | 均值滤波 | **1709.14** | **15.803** | **0.6316** | +36.374 |
| 高斯噪声 | 中值滤波 | 2122.36 | 14.863 | 0.5494 | +41.480 |
| 高斯噪声 | 高斯滤波 | 1738.98 | 15.728 | 0.5376 | +36.378 |

**椒盐噪声分析。** 在本图与本次参数下，中值滤波的三项相似性指标均最好，PSNR 从 `8.886 dB` 提升到 `31.221 dB`。椒盐噪声形成极端通道值，中值操作可以抑制离群值，同时较好保留人物轮廓、发丝和花束边缘。均值、高斯滤波也降低了误差，但会将极端值参与邻域平均，残留偏差并模糊细节。

**高斯噪声分析。** 本次 5×5 均值滤波的 MSE 最低、PSNR 和 SSIM 最高；高斯滤波的 PSNR 接近均值滤波，但 SSIM 较低。不能仅凭噪声名称认定高斯滤波必然最优。结果取决于图像、噪声参数、窗口和评价标准，此处结论仅对应本次设置。三种方法都提高了 PSNR 和 SSIM，但正均值噪声带来的偏亮仍然存在：均值、高斯滤波后的平均误差约为 `+36.37`，中值滤波约为 `+41.48`。平滑操作能削弱随机波动，无法自动消除系统性偏移，也无法恢复裁剪前已经丢失的高亮信息。

**实现一致性。** 本次运行通过 BGR→RGB→BGR 无损往返检查、滤波输出形状与 `uint8` 类型检查，以及手动中值与 OpenCV 全图逐元素一致性检查。手动结果可以复现 OpenCV 在本输入、窗口和边界条件下的中值滤波输出。

完整精度的数值和运行参数见 [metrics.csv](result/metrics.csv) 与 [summary.json](result/summary.json)。摘要记录了输入文件 SHA-256、像素值、随机种子、依赖版本和一致性检查结果，便于核对复现输入。

## 实验小结：

<p style="text-indent: 2em;">本次实验使用用户提供的完整原图，完成了图像读取、BGR/RGB 通道顺序转换、灰度化、椒盐与高斯加噪、三种滤波对比及手动中值滤波。结果表明，中值滤波对本次椒盐噪声最有效；本次高斯噪声下，均值滤波取得了三种方法中更好的指标，但不能消除正均值噪声及裁剪造成的失真。手动中值滤波与 OpenCV 输出完全一致，验证了邻域中位数计算与复制边界处理的实现。</p>

## 复现运行：

从仓库根目录执行以下 PowerShell 命令。已存在可用的 `.venv` 时可直接使用该环境。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r exp2/requirements.txt
.venv\Scripts\python.exe exp2/experiment.py
```

默认读取 `exp2/assets/input.jpg`，将结果写入 `exp2/result/`。脚本固定椒盐噪声随机种子为 `2026`、高斯噪声随机种子为 `2027`，窗口大小为 `5`；直接依赖版本由 [requirements.txt](requirements.txt) 固定。

在支持 Jupyter Notebook 的编辑器中打开 [code.ipynb](code.ipynb)，选择 `.venv` 中的 Python 解释器作为内核，按顺序运行所有单元，也可完成同一实验。Notebook 与脚本共用算法实现，结果保存到相同目录。
