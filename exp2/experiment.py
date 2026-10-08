"""实验二：使用用户图片完成颜色转换、加噪与滤波。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise


BASE_DIR = Path(__file__).resolve().parent


def read_image(path: Path) -> np.ndarray:
    """通过 imdecode 读取中文路径，返回 OpenCV 的 BGR uint8 图像。"""
    encoded = Path(path).read_bytes()
    if not encoded:
        raise ValueError(f"图像文件为空：{path}")
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图像：{path}")
    return image


def validate_kernel(kernel_size: int) -> None:
    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("滤波窗口必须是大于等于 3 的奇数。")


def manual_median_filter_color(
    image: np.ndarray, kernel_size: int = 5, block_rows: int = 64
) -> np.ndarray:
    """逐通道取邻域中值，不调用任何现成滤波函数。

    edge 填充对应 OpenCV 中值滤波的 BORDER_REPLICATE。
    将连续若干行的邻域组织成数组，批量求中值；与逐像素循环的
    数学过程相同，分块可控制全分辨率图片的中间内存占用。
    """
    validate_kernel(kernel_size)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入必须是 H×W×3 的 uint8 彩色图像。")
    if image.shape[0] == 0 or image.shape[1] == 0 or block_rows < 1:
        raise ValueError("图像不能为空，block_rows 必须大于 0。")

    height = image.shape[0]
    pad = kernel_size // 2
    result = np.empty_like(image)
    for channel_index in range(3):
        channel = np.pad(image[:, :, channel_index], pad, mode="edge")
        for start in range(0, height, block_rows):
            end = min(start + block_rows, height)
            windows = sliding_window_view(
                channel[start : end + 2 * pad, :], (kernel_size, kernel_size)
            )
            result[start:end, :, channel_index] = np.median(windows, axis=(-2, -1))
    return result


def to_uint8(image: np.ndarray) -> np.ndarray:
    """random_noise 的 [0, 1] 浮点输出统一四舍五入到 uint8。"""
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def quality_metrics(reference: np.ndarray, image: np.ndarray) -> dict:
    difference = image.astype(np.float64) - reference.astype(np.float64)
    return {
        "mse": float(np.mean(difference**2)),
        "psnr_db": float(peak_signal_noise_ratio(reference, image, data_range=255)),
        "ssim": float(structural_similarity(reference, image, data_range=255, channel_axis=2)),
        "mean_error": float(np.mean(difference)),
    }


def save_comparison(
    path: Path, panels: list[tuple[np.ndarray, str]], rows: int, columns: int
) -> None:
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 6 * rows), squeeze=False)
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=255)
        axis.set_title(title, fontsize=11, pad=10)
        axis.axis("off")
    # 为两行标题与上下行之间预留固定空隙，避免等比例图像挤压标题。
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.025, top=0.92 if rows == 1 else 0.96,
                           wspace=0.08, hspace=0.14)
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def write_image(path: Path, rgb_or_gray: np.ndarray) -> None:
    image = cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2BGR) if rgb_or_gray.ndim == 3 else rgb_or_gray
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"图像编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def run_experiment(
    input_path: Path, output_dir: Path, seed: int = 2026, kernel_size: int = 5
) -> dict:
    """在完整原图上运行，输出正式结果图、指标和实验摘要。"""
    validate_kernel(kernel_size)
    if seed < 0:
        raise ValueError("随机种子必须大于等于 0。")
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = read_image(input_path)
    if min(bgr.shape[:2]) <= 100:
        raise ValueError("图像宽高必须大于 100，以读取 [100,100] 像素。")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    write_image(output_dir / "gray_image.png", gray)
    save_comparison(
        output_dir / "color_spaces.png",
        [(bgr, "BGR shown as RGB (incorrect)"), (rgb, "RGB (correct)"), (gray, "Grayscale")],
        1, 3,
    )

    # 保留参考仓库的噪声参数；两类噪声使用各自固定的随机数流。
    sp_float = random_noise(rgb, mode="s&p", amount=0.4, rng=np.random.default_rng(seed))
    gaussian_float = random_noise(
        rgb, mode="gaussian", mean=0.2, var=0.03, rng=np.random.default_rng(seed + 1)
    )
    noises = {"salt_pepper": to_uint8(sp_float), "gaussian": to_uint8(gaussian_float)}
    labels = {"salt_pepper": "S&P", "gaussian": "Gaussian"}
    save_comparison(
        output_dir / "noise_comparison.png",
        [(rgb, "Original RGB"), (noises["salt_pepper"], "S&P: amount=0.4"),
         (noises["gaussian"], "Gaussian: mean=0.2, var=0.03")],
        1, 3,
    )

    filters = {}
    records = []
    for noise_name, noisy in noises.items():
        # 三种方法使用同一份 uint8 输入、同样大小的窗口和复制边界。
        filters[noise_name] = {
            "mean": cv2.blur(noisy, (kernel_size, kernel_size), borderType=cv2.BORDER_REPLICATE),
            "median": cv2.medianBlur(noisy, kernel_size),
            "gaussian": cv2.GaussianBlur(
                noisy, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_REPLICATE
            ),
        }
        for method, image in [("noisy", noisy), *filters[noise_name].items()]:
            records.append({"noise": noise_name, "method": method, **quality_metrics(rgb, image)})

    metric_lookup = {(record["noise"], record["method"]): record for record in records}
    panels = []
    for noise_name in noises:
        for method, image in filters[noise_name].items():
            metrics = metric_lookup[noise_name, method]
            title = (f"{labels[noise_name]} + {method.title()} ({kernel_size}x{kernel_size})\n"
                     f"PSNR {metrics['psnr_db']:.2f} dB | SSIM {metrics['ssim']:.4f}")
            panels.append((image, title))
    save_comparison(output_dir / "filter_results_2x3.png", panels, 2, 3)

    manual = manual_median_filter_color(noises["salt_pepper"], kernel_size)
    opencv_median = filters["salt_pepper"]["median"]
    absolute_difference = np.abs(manual.astype(np.int16) - opencv_median.astype(np.int16))
    validation = {
        "bgr_rgb_roundtrip": bool(np.array_equal(bgr, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))),
        "manual_matches_opencv": bool(np.array_equal(manual, opencv_median)),
        "manual_max_abs_difference": int(absolute_difference.max()),
        "manual_different_channel_values": int(np.count_nonzero(absolute_difference)),
        "output_shape_preserved": bool(all(
            image.shape == rgb.shape and image.dtype == np.uint8
            for methods in filters.values() for image in methods.values()
        ) and manual.shape == rgb.shape and manual.dtype == np.uint8),
    }
    if not all(validation[key] for key in
               ("bgr_rgb_roundtrip", "manual_matches_opencv", "output_shape_preserved")):
        raise RuntimeError(f"实验一致性检查未通过：{validation}")
    save_comparison(
        output_dir / "manual_median.png",
        [(noises["salt_pepper"], "S&P input"), (opencv_median, "OpenCV median"),
         (manual, f"Manual median\nMax difference: {absolute_difference.max()}")],
        1, 3,
    )
    write_image(output_dir / "denoised_sp.png", manual)

    summary = {
        "input": {
            "file": input_path.name,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "channels": 3,
            "dtype": str(rgb.dtype), "pixel_row_col": [100, 100],
            "pixel_bgr": bgr[100, 100].tolist(), "pixel_rgb": rgb[100, 100].tolist(),
            "pixel_gray": int(gray[100, 100]),
        },
        "parameters": {
            "seed_salt_pepper": seed, "seed_gaussian": seed + 1,
            "salt_pepper_amount": 0.4, "salt_vs_pepper": 0.5,
            "gaussian_mean": 0.2, "gaussian_variance": 0.03,
            "kernel_size": kernel_size, "gaussian_sigma": 0,
            "border": "BORDER_REPLICATE", "quantization": "round_to_nearest_uint8",
        },
        "noise_observations": {
            "salt_pepper_changed_channel_fraction": float(np.mean(noises["salt_pepper"] != rgb)),
            "salt_pepper_changed_pixel_fraction": float(np.mean(np.any(noises["salt_pepper"] != rgb, axis=2))),
            "gaussian_upper_clipped_channel_fraction": float(np.mean(gaussian_float == 1)),
            "gaussian_lower_clipped_channel_fraction": float(np.mean(gaussian_float == 0)),
        },
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__,
                     "matplotlib": matplotlib.__version__, "scikit_image": skimage.__version__},
        "metrics": records,
        "validation": validation,
    }
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_DIR / "assets" / "input.jpg")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "result")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--kernel-size", type=int, default=5)
    arguments = parser.parse_args()
    try:
        summary = run_experiment(arguments.input, arguments.output, arguments.seed, arguments.kernel_size)
    except (OSError, ValueError, cv2.error, RuntimeError) as error:
        parser.exit(1, f"实验运行失败：{error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
