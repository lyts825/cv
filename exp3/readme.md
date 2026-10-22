# 实验三：图像关键点检测及图像拼接

实验日期：2026-10-22

本实验按照 [参考实验](https://github.com/Guyudong1/CV_exp/tree/main/3rd.exp) 的流程，使用给定插画完成关键点检测、特征匹配和图像拼接。原图见 [input.jpg](assets/input.jpg)，源码见 [experiment.py](experiment.py)，分步运行与结果展示见 [code.ipynb](code.ipynb)。

从同一张插画裁出两幅重叠视图，再分别模拟尺寸、亮度和透视变化。实验用于验证合成视图的配准流程，真实拍摄中的视差、运动和遮挡需要另行验证。

## 实验目的

学习 SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）的关键点检测与描述方法。用 FLANN（Fast Library for Approximate Nearest Neighbors，近似最近邻检索库）匹配描述符。再用 RANSAC（Random Sample Consensus，随机采样一致性）估计单应矩阵，将图像对齐并融合。

## 素材与实验设置

原图尺寸为 `3268×5765`，保存时保留原始文件字节。处理时保持宽高比，将长边缩至 1800 像素，得到 `1020×1800` 的工作图。后续几何误差均以视图 A 在工作分辨率下的像素为单位。

从工作图裁出视图 A 和 B。数组索引从 0 开始，裁剪范围不含右边界。

```python
image_a = working[:, 0:734]
image_b = working[:, 286:1020]
```

A、B 的尺寸均为 `734×1800`。重叠宽度为 `734-286=448` 像素，占单幅视图宽度的 `61.04%`。原图尺寸、裁剪坐标和文件的 SHA-256 摘要见 [summary.json](result/summary.json)。

| 实验组 | A | B | 目的 |
| --- | --- | --- | --- |
| 重叠裁剪 `baseline` | 左侧裁剪视图 | 右侧裁剪视图 | 检查基础匹配和水平配准 |
| 尺寸变化 `scale` | 保持原尺寸 | 宽高缩至 75%，得到 `550×1350` | 检查尺度变化后的匹配 |
| 亮度变化 `brightness` | 保持原亮度 | `round(0.72*B+12)`，裁剪到 `[0,255]` | 比较亮度变化及补偿效果 |
| 透视变化 `perspective` | 保持原视图 | 按已知四角映射生成透视图 | 检查透视单应估计 |

透视组将 B 的四角映射到给定位置，按左上、右上、右下、左下排序。目标坐标依次为 `(0.045w,0.025h)`、`(0.97w,0)`、`(0.995w,0.965h)` 和 `(0.015w,0.995h)`，其中 `w=734`、`h=1800`。输入视图和透视组的有效掩膜保存在 [img](img)。

四组共用以下参数。OpenCV 使用单线程，各组随机种子依次为 `2026`、`2027`、`2028` 和 `2029`。

| 步骤 | 参数 |
| --- | --- |
| SIFT | `cv2.SIFT_create()` 默认参数 |
| FLANN | KDTree，`trees=5`、`checks=50`、`k=2` |
| Lowe 比率筛选 | 最近邻距离小于次近邻距离的 `0.7` 倍 |
| RANSAC | 重投影阈值 `5.0` 像素，估计 B→A 的单应矩阵 |
| 亮度补偿 | 在有效重叠区按通道拟合 `A≈gain*B+bias` |
| 图像融合 | 按有效区域边界距离加权羽化 |

## 实验过程

### 1. 读取图像与灰度转换

使用 `cv2.imdecode` 解码文件字节，支持含中文的路径。OpenCV 返回 BGR 图像，绘图时转换为 RGB，特征检测时转换为灰度图。

```python
gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
```

下图依次为彩色 A、彩色 B、灰度 A 和灰度 B。

![两幅输入视图及其灰度图](result/input_views.png)

### 2. SIFT 关键点检测与描述

SIFT 在灰度图上检测关键点，并为每个关键点计算描述符。基准组中，A 检测到 `1866` 个关键点，B 检测到 `1889` 个关键点。图中圆圈和方向线表示关键点的尺度与主方向。

```python
sift = cv2.SIFT_create()
keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, detection_mask)
```

透视图和有效掩膜使用相同的线性插值方式生成，掩膜只保留完全覆盖的像素。检测前再将掩膜内缩，排除黑色填充及混合边界产生的特征。

![基准组 SIFT 关键点](result/sift_keypoints.png)

### 3. FLANN 匹配与 Lowe 比率筛选

FLANN 为 A 中的每个描述符检索 B 中的两个近邻。最近邻与次近邻的距离越接近，匹配越难区分。Lowe 比率筛选保留最近邻距离与次近邻距离之比小于 `0.7` 的匹配。

```python
matcher = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=5), dict(checks=50)
)
pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
good = [pair[0] for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
```

基准组共得到 `1866` 组近邻检索结果，筛选后保留 `1159` 对匹配。SIFT 负责检测关键点和计算描述符，FLANN 负责匹配描述符。

### 4. RANSAC 单应矩阵估计

从筛选后的匹配中提取 B、A 的点坐标，估计从 B 到 A 的单应矩阵。RANSAC 将符合该几何模型的匹配标为内点。基准组得到 `1135` 对内点，占优质匹配的 `97.93%`。

```python
points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good])
points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good])
H, mask = cv2.findHomography(points_b, points_a, cv2.RANSAC, 5.0)
```

基准组估计的矩阵如下。横向平移接近裁剪偏移 `286` 像素，其余项接近单位矩阵的对应项。

```text
[[ 0.99998983, -0.00001263, 286.00770244],
 [-0.00001465,  1.00000853,  -0.00232062],
 [-0.00000002,  0.00000000,   1.00000000]]
```

匹配图将 RANSAC 内点按描述符距离从小到大排序，最多显示 80 对，避免连线过密。全部匹配数和内点数见下方指标表。

![基准组内点匹配示意](result/matches_baseline.jpg)

### 5. 画布平移与图像融合

程序先将 B 的四角映射到 A 的坐标系，再与 A 的四角一起确定画布范围。最小坐标向下取整，最大坐标向上取整。平移矩阵 `T` 将负坐标移入画布，B 使用 `T @ H` 映射到画布中。

```python
lower = np.floor(corners.min(axis=0)).astype(int)
upper = np.ceil(corners.max(axis=0)).astype(int)
width, height = (upper - lower + 1).tolist()
T = np.array([[1., 0., -lower[0]],
              [0., 1., -lower[1]],
              [0., 0., 1.]])
warped_b = cv2.warpPerspective(image_b, T @ H, (width, height))
```

程序用掩膜确定有效区域，保留原图中的黑色内容像素。将线性插值后完全有效的重叠区向内收缩 2 像素，再用于亮度拟合和误差评价。

融合前，程序按颜色通道拟合 `A≈gain*B+bias`，筛除残差较大的样本后再次拟合。补偿后的 B 与 A 按像素到各自有效区域边界的距离加权，距离越大，权重越高。

参考实验采用直接覆盖，本实验增加了羽化和亮度补偿。亮度组以直接覆盖作对照，下图从左到右依次为直接覆盖、仅羽化和亮度补偿后羽化。

![亮度变化组的三种融合方法](result/brightness_comparison.png)

## 实验结果与分析

下表列出本次运行的实测结果。内点率为 RANSAC 内点数除以 Lowe 筛选后的优质匹配数。

| 实验组 | A 关键点 | B 关键点 | 优质匹配 | RANSAC 内点 | 内点率 | 内点 RMSE / px | 角点 RMSE / px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 重叠裁剪 | 1866 | 1889 | 1159 | 1135 | 97.93% | 0.1151 | 0.0174 |
| 尺寸变化 | 1866 | 1387 | 548 | 524 | 95.62% | 0.5148 | 0.3273 |
| 亮度变化 | 1866 | 1308 | 751 | 715 | 95.21% | 0.1928 | 0.0476 |
| 透视变化 | 1866 | 1928 | 732 | 704 | 96.17% | 0.4948 | 0.2921 |

RMSE（Root Mean Square Error，均方根误差）先计算二维坐标误差的欧氏距离，再求距离的平方均值并开方。内点 RMSE 比较 RANSAC 内点经估计矩阵映射后的坐标与 A 中的匹配坐标。角点 RMSE 比较 B 的四个矩形角点分别经估计矩阵和已知变换映射后的坐标。两项误差均以 A 在工作分辨率下的像素为单位。

已知裁剪和变换矩阵仅用于计算角点误差。特征匹配、RANSAC 和拼接使用从图像估计的结果。尺寸组的已知变换同时计入实际缩放比例和 OpenCV 缩放的像素中心偏移。

| 实验组 | 补偿前重叠 MAE | 补偿后重叠 MAE | 拼接尺寸（宽×高） |
| --- | ---: | ---: | --- |
| 重叠裁剪 | 0.004 | 0.004 | `1021×1802` |
| 尺寸变化 | 1.437 | 1.437 | `1020×1801` |
| 亮度变化 | 38.147 | 0.270 | `1021×1802` |
| 透视变化 | 1.000 | 1.000 | `1043×1914` |

MAE（Mean Absolute Error，平均绝对误差）在内缩后的有效重叠区计算，取所有 BGR 通道绝对差的平均值，通道数值范围为 `[0,255]`。该指标描述对齐后 A 与 B 的像素值差异。

基准组的角点误差为 `0.0174 px`，估计变换接近已知水平平移。尺寸组中，B 缩小后，关键点数从 `1889` 减少至 `1387`，仍得到 `524` 对内点。缩放和再次变换产生的插值误差，使重叠 MAE 高于基准组。

亮度组中，B 的关键点数降至 `1308`，仍得到 `715` 对内点。配准后直接覆盖会留下可见的明暗接缝，仅羽化会将明暗变化扩散到重叠区。加入亮度补偿后，重叠 MAE 从 `38.147` 降至 `0.270`。这一结果对应本次全局仿射亮度变化。

透视组得到 `704` 对内点，角点误差为 `0.2921 px`。图中黑边来自透视变换后的画布空白，掩膜将这些区域排除在匹配和重叠指标之外。其他组也可能因估计误差和边界取整而多出少量边缘像素。

下图每行依次为输入 B、内点匹配示意和最终拼接结果。四行依次对应重叠裁剪、尺寸变化、亮度变化和透视变化。

![四组实验总览](result/overview.png)

各组完整拼接图：[重叠裁剪](result/panorama_baseline.png)、[尺寸变化](result/panorama_scale.png)、[亮度变化](result/panorama_brightness.png) 和 [透视变化](result/panorama_perspective.png)。完整数据见 [metrics.csv](result/metrics.csv) 和 [summary.json](result/summary.json)。

## 验证与适用范围

脚本完成四组实验，五项内置检查均通过：各组至少有 4 对内点，内点率超过 80%，角点 RMSE 和内点 RMSE 均低于 `2 px`，亮度组补偿后的重叠 MAE 下降。这些阈值只用于本次合成实验，其他图像需单独检验。

Notebook 格式校验通过，从 `exp3` 目录顺序执行全部 9 个代码单元也通过。附件与所保存原图的 SHA-256 一致。精确重建、黑色内容像素保留、负坐标画布和空白图报错检查均通过，`pip check` 未发现依赖冲突。

执行副本和本地验证材料保存在 Git 忽略目录 `.local/`。正式 Notebook 保留源码，结果图单独保存。

两幅视图来自同一张插画，重叠内容一致。真实照片中的视差、镜头畸变、局部光照变化和动态物体需要另行验证。羽化无法消除几何错位，整幅拼接图的质量也需结合重叠 MAE 以外的证据判断。

本次运行环境为 Python `3.12.10`、OpenCV `4.13.0`、NumPy `2.4.4` 和 Matplotlib `3.10.8`。锁定安装版本见 [requirements.txt](requirements.txt)。

## 运行方法

在仓库根目录执行以下命令（Python 3.12，Windows PowerShell）。已有 `.venv` 时，可跳过创建环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r exp3/requirements.txt
.\.venv\Scripts\python.exe exp3/experiment.py
```

默认输入为 `exp3/assets/input.jpg`，派生视图写入 `exp3/img/`，结果写入 `exp3/result/`。自定义参数示例：

```powershell
.\.venv\Scripts\python.exe exp3/experiment.py --input exp3/assets/input.jpg --output exp3/result --long-edge 1800 --seed 2026
```

也可以从仓库根目录或 `exp3` 目录打开 [code.ipynb](code.ipynb)，选择 `.venv` 内核，依次运行单元格。Notebook 与脚本使用相同的算法代码，运行时会重新生成结果。

## 版本记录

`v0.1.1 → v0.2.0`（2026-10-22）：新增实验三源码、Notebook、用户原图、派生视图和四组实验结果，保留前两次实验入口。变更为兼容新增；脚本和 Notebook 验收通过。已知限制为单图模拟与透视画布空白，对应提交以 Git 标签 `v0.2.0` 定位。

`v0.2.0 → v0.2.1`（2026-10-22）：润色实验报告、Notebook 说明和仓库中的实验三介绍，精简重复表述，明确指标定义与适用范围。代码、实测数据和结论保持不变；文档差异与内容一致性检查通过。实验运行记录仍对应 `v0.2.0`，文档修订提交以 Git 标签 `v0.2.1` 定位。
