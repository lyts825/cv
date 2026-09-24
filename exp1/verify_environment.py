"""Run the package checks described in the experiment report."""

from __future__ import annotations

import argparse


def verify_pytorch() -> None:
    import torch

    result = torch.tensor([1, 2, 3, 4], dtype=torch.float32).sum().item()
    print("PyTorch verification")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Tensor sum: {result:.1f}")
    print(f"CUDA available: {torch.cuda.is_available()}")


def verify_opencv() -> None:
    import cv2
    import numpy as np

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (55, 55), (255, 255, 255), 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonzero = int(np.count_nonzero(image))
    print("OpenCV verification")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Image shape: {image.shape}")
    print(f"Gray image shape: {gray.shape}")
    print(f"Nonzero pixels after rectangle: {nonzero}")


def verify_sklearn() -> None:
    import sklearn
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    features, labels = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=3,
    )
    model = LogisticRegression(max_iter=1000).fit(features, labels)
    accuracy = model.score(features, labels)
    print("Scikit-learn verification")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"Samples: {len(features)}")
    print(f"Training accuracy: {accuracy:.3f}")


CHECKS = {
    "pytorch": verify_pytorch,
    "opencv": verify_opencv,
    "sklearn": verify_sklearn,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=[*CHECKS, "all"], default="all")
    args = parser.parse_args()

    if args.check == "all":
        for check in CHECKS.values():
            check()
        print("All imports: PASS")
        return

    CHECKS[args.check]()


if __name__ == "__main__":
    main()
