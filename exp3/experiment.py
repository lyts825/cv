"""实验三：用同图模拟视图完成 SIFT 检测、FLANN 匹配和图像拼接。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
VERSION = "0.2.0"
REFERENCE_COMMIT = "68ae1f5fa7cbb6b1cf96a981f35f46681357975d"
CASE_LABELS = {
    "baseline": "重叠裁剪",
    "scale": "尺寸变化",
    "brightness": "亮度变化",
    "perspective": "透视变化",
}


def read_image(path: Path) -> np.ndarray:
    """从字节解码 BGR 图像，支持中文路径。"""
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图像：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = [cv2.IMWRITE_JPEG_QUALITY, 93] if path.suffix.lower() == ".jpg" else []
    success, encoded = cv2.imencode(path.suffix, image, options)
    if not success:
        raise RuntimeError(f"图像编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def image_corners(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.float64([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2),
                                    homography).reshape(-1, 2)


def prepare_views(image: np.ndarray, long_edge: int = 1800) -> tuple[np.ndarray, list, dict]:
    """裁出约 61% 重叠的视图，并记录独立于特征估计的真实变换。"""
    if long_edge < 400:
        raise ValueError("处理长边至少为 400 像素。")
    height, width = image.shape[:2]
    ratio = min(1.0, long_edge / max(height, width))
    working = cv2.resize(image, (round(width * ratio), round(height * ratio)),
                         interpolation=cv2.INTER_AREA) if ratio < 1 else image.copy()
    height, width = working.shape[:2]
    if min(height, width) < 200:
        raise ValueError("缩放后的图像短边至少为 200 像素。")
    a_end, b_start = round(0.72 * width), round(0.28 * width)
    image_a, image_b = working[:, :a_end].copy(), working[:, b_start:].copy()
    translation = np.array([[1., 0., b_start], [0., 1., 0.], [0., 0., 1.]])
    cases = [{"id": "baseline", "image": image_b, "truth": translation,
              "description": "A=左侧72%，B=右侧72%；重叠为工作图宽度的约44%。"}]

    b_height, b_width = image_b.shape[:2]
    scaled = cv2.resize(image_b, (round(b_width * 0.75), round(b_height * 0.75)),
                        interpolation=cv2.INTER_AREA)
    sx, sy = scaled.shape[1] / b_width, scaled.shape[0] / b_height
    # OpenCV resize 的像素中心约定：x_dst = s * (x_src + 0.5) - 0.5。
    resize_matrix = np.array([[sx, 0., (sx - 1.) / 2.],
                              [0., sy, (sy - 1.) / 2.], [0., 0., 1.]])
    cases.append({"id": "scale", "image": scaled,
                  "truth": translation @ np.linalg.inv(resize_matrix),
                  "description": "仅将 B 的宽高缩至75%，保留 A 的尺寸。"})
    darker = np.rint(image_b.astype(np.float32) * 0.72 + 12.).clip(0, 255).astype(np.uint8)
    cases.append({"id": "brightness", "image": darker, "truth": translation,
                  "description": "仅将 B 变为 round(0.72*B+12)，逐通道裁剪到[0,255]。"})

    source = image_corners(image_b).astype(np.float32)
    destination = np.float32([[0.045 * b_width, 0.025 * b_height],
                              [0.97 * b_width, 0.],
                              [0.995 * b_width, 0.965 * b_height],
                              [0.015 * b_width, 0.995 * b_height]])
    perspective = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(image_b, perspective, (b_width, b_height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    coverage = cv2.warpPerspective(np.full((b_height, b_width), 255, np.uint8),
                                   perspective, (b_width, b_height), flags=cv2.INTER_LINEAR)
    mask = (coverage == 255).astype(np.uint8) * 255
    cases.append({"id": "perspective", "image": warped,
                  "mask": mask, "truth": translation @ np.linalg.inv(perspective),
                  "description": "对 B 施加已知四角透视变换，黑色填充由有效掩膜排除。"})
    for case in cases:
        case.setdefault("mask", np.full(case["image"].shape[:2], 255, np.uint8))
        case["label"] = CASE_LABELS[case["id"]]
    metadata = {"working_width": width, "working_height": height,
                "crop_a_xyxy": [0, 0, a_end, height],
                "crop_b_xyxy": [b_start, 0, width, height],
                "overlap_width": a_end - b_start,
                "overlap_fraction_of_a": (a_end - b_start) / a_end}
    return image_a, cases, metadata


def detect_and_match(image_a: np.ndarray, image_b: np.ndarray,
                     mask_b: np.ndarray | None = None, seed: int = 2026) -> dict:
    """SIFT → FLANN → Lowe 比率筛选 → RANSAC，估计 B 到 A 的映射。"""
    cv2.setRNGSeed(seed)
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # 排除透视视图的填充边缘，避免把黑边当作内容特征。
    detection_mask = cv2.erode(mask_b, np.ones((9, 9), np.uint8)) if mask_b is not None else None
    keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, detection_mask)
    if descriptors_a is None or descriptors_b is None or len(descriptors_b) < 2:
        raise ValueError("有效 SIFT 描述符不足，无法完成双近邻匹配。")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2
            and pair[0].distance < 0.7 * pair[1].distance]
    if len(good) < 4:
        raise ValueError(f"优质匹配仅 {len(good)} 对；单应矩阵至少需要4对。")
    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good])
    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good])
    homography, mask = cv2.findHomography(points_b, points_a, cv2.RANSAC, 5.0)
    if homography is None or mask is None or not np.isfinite(homography).all():
        raise ValueError("RANSAC 未能估计有效的单应矩阵。")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4 or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("单应矩阵退化或内点不足。")
    errors = np.linalg.norm(project_points(points_b, homography) - points_a, axis=1)
    return {"keypoints_a": keypoints_a, "keypoints_b": keypoints_b,
            "knn_pairs": len(pairs), "good": good, "inliers": inliers,
            "homography": homography / homography[2, 2],
            "inlier_rmse_px": float(np.sqrt(np.mean(errors[inliers] ** 2)))}


def correct_exposure(warped_a: np.ndarray, warped_b: np.ndarray,
                     overlap: np.ndarray) -> tuple[np.ndarray, list, list]:
    """在有效重叠区拟合 A≈gain*B+bias；同一组系数应用于整幅 B。"""
    locations = np.flatnonzero(overlap)
    if len(locations) < 100:
        raise ValueError("重叠像素不足，无法估计亮度补偿。")
    locations = locations[::max(1, len(locations) // 50000)]
    samples_a = warped_a.reshape(-1, 3)[locations].astype(np.float64)
    samples_b = warped_b.reshape(-1, 3)[locations].astype(np.float64)
    gains, biases = [], []
    for channel in range(3):
        x, y = samples_b[:, channel], samples_a[:, channel]
        design = np.column_stack([x, np.ones_like(x)])
        gain, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        # 一次残差筛选减小局部配准误差对全局亮度拟合的影响。
        residual = np.abs(gain * x + bias - y)
        keep = residual <= max(2., np.quantile(residual, 0.9))
        if keep.sum() >= 100 and np.std(x[keep]) > 1:
            gain, bias = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        gains.append(float(np.clip(gain, 0.25, 4.)))
        biases.append(float(np.clip(bias, -100., 100.)))
    corrected = np.rint(warped_b.astype(np.float32) * np.array(gains)
                         + np.array(biases)).clip(0, 255).astype(np.uint8)
    return corrected, gains, biases


def feather_blend(image_a: np.ndarray, image_b: np.ndarray,
                  valid_a: np.ndarray, valid_b: np.ndarray) -> np.ndarray:
    """按到有效域边界的距离加权，掩膜保留合法黑色像素。"""
    def distance(mask):
        padded = np.pad(mask.astype(np.uint8), 1)
        return cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]

    weight_a, weight_b = distance(valid_a), distance(valid_b)
    total = weight_a + weight_b
    alpha = np.divide(weight_a, total, out=np.zeros_like(total), where=total > 0)
    blend = image_a.astype(np.float32) * alpha[..., None]
    blend += image_b.astype(np.float32) * (1. - alpha[..., None])
    blend[~(valid_a | valid_b)] = 0
    return np.rint(blend).clip(0, 255).astype(np.uint8)


def stitch_pair(image_a: np.ndarray, image_b: np.ndarray, homography: np.ndarray,
                mask_b: np.ndarray | None = None) -> dict:
    """计算完整画布，变换有效掩膜，再输出直接覆盖和补偿羽化结果。"""
    corners = np.vstack([image_corners(image_a), project_points(image_corners(image_b), homography)])
    if not np.isfinite(corners).all():
        raise ValueError("图像角点映射无效。")
    lower = np.floor(corners.min(axis=0)).astype(int)
    upper = np.ceil(corners.max(axis=0)).astype(int)
    width, height = (upper - lower + 1).tolist()
    if width <= 0 or height <= 0 or width * height > 8 * (image_a.size + image_b.size) / 3:
        raise ValueError("单应变换产生异常大的画布。")
    translation = np.array([[1., 0., -lower[0]], [0., 1., -lower[1]], [0., 0., 1.]])
    size = (width, height)
    warped_a = cv2.warpPerspective(image_a, translation, size)
    warped_b = cv2.warpPerspective(image_b, translation @ homography, size)
    mask_a = np.full(image_a.shape[:2], 255, np.uint8)
    mask_b = np.full(image_b.shape[:2], 255, np.uint8) if mask_b is None else mask_b
    # 使用线性插值后的满覆盖区域，排除填充与内容混合的边界像素。
    valid_a = cv2.warpPerspective(mask_a, translation, size) == 255
    valid_b = cv2.warpPerspective(mask_b, translation @ homography, size) == 255
    if not (valid_a & valid_b).any():
        raise ValueError("变换后没有有效重叠区域。")
    overlap = valid_a & valid_b
    # 评价用内缩重叠区，减小插值填充边缘的影响。
    inner_overlap = cv2.erode(overlap.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    corrected_b, gains, biases = correct_exposure(warped_a, warped_b, inner_overlap)
    hard = warped_b.copy()
    hard[valid_a] = warped_a[valid_a]
    hard[~(valid_a | valid_b)] = 0
    raw_blend = feather_blend(warped_a, warped_b, valid_a, valid_b)
    panorama = feather_blend(warped_a, corrected_b, valid_a, valid_b)
    raw_mae = np.abs(warped_a[inner_overlap].astype(float) - warped_b[inner_overlap]).mean()
    corrected_mae = np.abs(warped_a[inner_overlap].astype(float) - corrected_b[inner_overlap]).mean()
    return {"panorama": panorama, "hard": hard, "raw_blend": raw_blend,
            "valid_mask": (valid_a | valid_b).astype(np.uint8) * 255,
            "translation": translation, "bounds_xyxy": [*lower.tolist(), *upper.tolist()],
            "overlap_pixels": int(inner_overlap.sum()), "overlap_mae_raw": float(raw_mae),
            "overlap_mae_corrected": float(corrected_mae),
            "exposure_gain_bgr": gains, "exposure_bias_bgr": biases}


def save_panels(path: Path, panels: list, rows: int, columns: int,
                figsize: tuple | None = None) -> None:
    figure, axes = plt.subplots(rows, columns, figsize=figsize or (4 * columns, 6 * rows),
                                squeeze=False)
    for axis, (image, title) in zip(axes.flat, panels):
        if image.ndim == 3:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title, fontsize=10, pad=8)
        axis.axis("off")
    for axis in list(axes.flat)[len(panels):]:
        axis.axis("off")
    figure.tight_layout(pad=1.5)
    figure.savefig(path, dpi=140, facecolor="white")
    plt.close(figure)


def run_experiment(input_path: Path, output_dir: Path,
                   long_edge: int = 1800, seed: int = 2026) -> dict:
    """执行四组模拟实验；已知变换仅用于评价，不参与匹配或拼接。"""
    if not 0 <= seed <= 2**31 - 1:
        raise ValueError("随机种子必须在0到2^31-1之间。")
    cv2.setNumThreads(1)
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(input_path)
    image_a, cases, view_metadata = prepare_views(image, long_edge)
    # 派生输入是正式可复现实验材料，放在输出目录旁的 img 中。
    view_dir = output_dir.parent / "img" if output_dir.name == "result" else output_dir / "img"
    view_dir.mkdir(parents=True, exist_ok=True)
    write_image(view_dir / "a.png", image_a)
    save_panels(output_dir / "input_views.png", [
        (image_a, "View A (BGR converted to RGB for display)"),
        (cases[0]["image"], "View B (overlap crop)"),
        (cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY), "Grayscale A"),
        (cv2.cvtColor(cases[0]["image"], cv2.COLOR_BGR2GRAY), "Grayscale B")], 1, 4)
    records, overview = [], []
    for index, case in enumerate(cases):
        case_id, image_b = case["id"], case["image"]
        write_image(view_dir / ("b.png" if case_id == "baseline" else f"b_{case_id}.png"), image_b)
        if case_id == "perspective":
            write_image(view_dir / "b_perspective_mask.png", case["mask"])
        match = detect_and_match(image_a, image_b, case["mask"], seed + index)
        homography = match["homography"]
        stitched = stitch_pair(image_a, image_b, homography, case["mask"])
        corners = image_corners(image_b)
        corner_error = project_points(corners, homography) - project_points(corners, case["truth"])
        corner_rmse = float(np.sqrt(np.mean(np.sum(corner_error**2, axis=1))))
        record = {"id": case_id, "label": case["label"], "description": case["description"],
                  "width_a": image_a.shape[1], "height_a": image_a.shape[0],
                  "width_b": image_b.shape[1], "height_b": image_b.shape[0],
                  "keypoints_a": len(match["keypoints_a"]), "keypoints_b": len(match["keypoints_b"]),
                  "knn_pairs": match["knn_pairs"], "good_matches": len(match["good"]),
                  "inliers": int(match["inliers"].sum()),
                  "inlier_ratio": float(match["inliers"].mean()),
                  "inlier_rmse_px": match["inlier_rmse_px"], "corner_rmse_px": corner_rmse,
                  "homography_b_to_a": homography.tolist(),
                  "expected_homography_b_to_a": case["truth"].tolist(),
                  "panorama_width": stitched["panorama"].shape[1],
                  "panorama_height": stitched["panorama"].shape[0]}
        for key in ("overlap_pixels", "overlap_mae_raw", "overlap_mae_corrected",
                    "exposure_gain_bgr", "exposure_bias_bgr", "bounds_xyxy"):
            record[key] = stitched[key]
        # 按描述符距离选择最多80条RANSAC内点，图例明确区分显示数与总数。
        inlier_matches = [m for m, valid in zip(match["good"], match["inliers"]) if valid]
        shown = sorted(inlier_matches, key=lambda m: m.distance)[:80]
        record["displayed_inlier_matches"] = len(shown)
        matching_image = cv2.drawMatches(image_a, match["keypoints_a"], image_b,
                                         match["keypoints_b"], shown, None,
                                         matchColor=(50, 190, 30),
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        write_image(output_dir / f"matches_{case_id}.jpg", matching_image)
        write_image(output_dir / f"panorama_{case_id}.png", stitched["panorama"])
        if case_id == "baseline":
            rich_a = cv2.drawKeypoints(image_a, match["keypoints_a"], None,
                                       color=(20, 160, 240), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            rich_b = cv2.drawKeypoints(image_b, match["keypoints_b"], None,
                                       color=(20, 160, 240), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            save_panels(output_dir / "sift_keypoints.png",
                        [(rich_a, f"A: {record['keypoints_a']} SIFT keypoints"),
                         (rich_b, f"B: {record['keypoints_b']} SIFT keypoints")], 1, 2)
        if case_id == "brightness":
            save_panels(output_dir / "brightness_comparison.png", [
                (stitched["hard"], "Direct overwrite (reference method)"),
                (stitched["raw_blend"], "Feather blend, no exposure correction"),
                (stitched["panorama"], "Exposure correction + feather blend")], 1, 3)
        overview.extend([(image_b, f"{case_id.title()}: input B"),
                         (matching_image, f"{record['inliers']}/{record['good_matches']} inliers; "
                          f"show {len(shown)}"),
                         (stitched["panorama"], f"Corner RMSE: {corner_rmse:.3f} px")])
        records.append(record)
    save_panels(output_dir / "overview.png", overview, 4, 3, figsize=(12, 18))
    validation = {
        "all_cases_have_at_least_four_inliers": all(r["inliers"] >= 4 for r in records),
        "all_inlier_ratios_above_80_percent": all(r["inlier_ratio"] > 0.8 for r in records),
        "all_corner_rmse_below_2px": all(r["corner_rmse_px"] < 2. for r in records),
        "all_inlier_rmse_below_2px": all(r["inlier_rmse_px"] < 2. for r in records),
        "brightness_overlap_mae_reduced": records[2]["overlap_mae_corrected"] < records[2]["overlap_mae_raw"],
    }
    summary = {
        "version": VERSION,
        "experiment_date": "2026-10-22",
        "reference": {"url": "https://github.com/Guyudong1/CV_exp/tree/main/3rd.exp",
                      "commit": REFERENCE_COMMIT},
        "input": {"file": input_path.name, "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  "width": image.shape[1], "height": image.shape[0], **view_metadata},
        "parameters": {"seed": seed, "case_seeds": [seed + i for i in range(4)],
                       "long_edge": long_edge, "ratio_threshold": 0.7,
                       "ransac_threshold_px": 5.0, "flann_trees": 5, "flann_checks": 50,
                       "scale_b": 0.75, "brightness_alpha": 0.72, "brightness_beta": 12,
                       "exposure_model": "per-channel robust affine fit on overlap",
                       "blend": "distance-transform feather", "opencv_threads": 1},
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "cases": records, "validation": validation,
        "limitations": ["从单张插画模拟重叠视图，不能评价真实拍摄的视差、运动或遮挡。",
                        "几何误差单位为A的工作分辨率像素；已知变换仅用于评价。",
                        "无独立多视角真值；重叠MAE仅描述两幅对齐图之间的亮度差。"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    fields = ["id", "keypoints_a", "keypoints_b", "knn_pairs", "good_matches", "inliers",
              "inlier_ratio", "inlier_rmse_px", "corner_rmse_px", "overlap_pixels",
              "overlap_mae_raw", "overlap_mae_corrected", "panorama_width", "panorama_height"]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_DIR / "assets" / "input.jpg")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "result")
    parser.add_argument("--long-edge", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    try:
        summary = run_experiment(arguments.input, arguments.output, arguments.long_edge, arguments.seed)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"实验运行失败：{error}\n")
    for case in summary["cases"]:
        print(f"{case['label']}: {case['good_matches']} 对优质匹配，{case['inliers']} 对内点，"
              f"角点RMSE={case['corner_rmse_px']:.4f} px，"
              f"重叠MAE={case['overlap_mae_raw']:.3f} → {case['overlap_mae_corrected']:.3f}")
    print("验收检查：", summary["validation"])
    print(f"结果目录：{arguments.output.resolve()}")
    if not all(summary["validation"].values()):
        parser.exit(1, "至少一项验收未通过，请检查输出指标。\n")


if __name__ == "__main__":
    main()
