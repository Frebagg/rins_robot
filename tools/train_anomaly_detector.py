#!/usr/bin/env python3
"""
Train a PaDiM anomaly detector using anomalib.

PaDiM fits a multivariate Gaussian to backbone patch features from OK images —
no gradient training, single pass, strong performance on industrial textures.

Usage:
    pip install anomalib
    python3 tools/train_anomaly_detector.py \
        --dataset /path/to/RINS_AD_dataset \
        --output  models/anomaly_model.pt

Requirements:
    anomalib >= 1.0
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile


def _organise(dataset_root: str, tmpdir: str) -> None:
    """
    Create an anomalib 2.x Folder-compatible layout inside *tmpdir*.

    anomalib 2.x Folder expects:
        <root>/normal/     ← OK images  (training)
        <root>/abnormal/   ← all damaged images  (anomalib splits val/test)
        <root>/mask/       ← GT masks matching abnormal/

    Test images from the dataset are merged into abnormal/ so anomalib
    can evaluate them as part of its test split.
    """
    train_img = os.path.join(dataset_root, 'train', 'images')
    train_gt  = os.path.join(dataset_root, 'train', 'gt')
    test_img  = os.path.join(dataset_root, 'test',  'images')
    test_gt   = os.path.join(dataset_root, 'test',  'gt')

    for d in ['normal', 'abnormal', 'mask']:
        os.makedirs(os.path.join(tmpdir, d), exist_ok=True)

    n_ok = n_dmg = 0
    for f in sorted(os.listdir(train_img)):
        if not f.endswith('.png'):
            continue
        src = os.path.abspath(os.path.join(train_img, f))
        if f.startswith('okay_'):
            os.symlink(src, os.path.join(tmpdir, 'normal', f))
            n_ok += 1
        elif f.startswith('damaged_'):
            os.symlink(src, os.path.join(tmpdir, 'abnormal', f))
            gt = os.path.abspath(os.path.join(train_gt, f))
            if os.path.exists(gt):
                os.symlink(gt, os.path.join(tmpdir, 'mask', f))
            n_dmg += 1

    # Merge test images into abnormal/ (prefix to avoid name collisions)
    n_test = 0
    for f in sorted(os.listdir(test_img)):
        if not f.endswith('.png'):
            continue
        src = os.path.abspath(os.path.join(test_img, f))
        dst_name = f'test_{f}'
        os.symlink(src, os.path.join(tmpdir, 'abnormal', dst_name))
        gt = os.path.abspath(os.path.join(test_gt, f))
        if os.path.exists(gt):
            os.symlink(gt, os.path.join(tmpdir, 'mask', dst_name))
        n_test += 1

    print(f'Dataset  OK={n_ok}  Damaged={n_dmg}  Test={n_test}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train PatchCore anomaly detector via anomalib')
    parser.add_argument('--dataset',    required=True,
                        help='Root of RINS_AD_dataset')
    parser.add_argument('--output',     default='models/anomaly_model.pt',
                        help='Output path for the exported .pt model')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--backbone',   default='resnet18')
    args = parser.parse_args()

    try:
        import anomalib
        print(f'anomalib {anomalib.__version__}')
    except ImportError:
        print('anomalib not found.  Install with:  pip install anomalib')
        return

    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Padim

    try:
        from anomalib.deploy import ExportType
    except ImportError:
        from anomalib.deploy.export import ExportType

    tmpdir = tempfile.mkdtemp(prefix='rins_ad_')
    try:
        _organise(args.dataset, tmpdir)

        from torchvision.transforms import v2 as T
        resize = T.Resize((args.image_size, args.image_size), antialias=True)

        datamodule = Folder(
            name='rins_tiles',
            root=tmpdir,
            normal_dir='normal',
            abnormal_dir='abnormal',
            mask_dir='mask',          # presence implies segmentation task
            train_batch_size=32,
            eval_batch_size=32,
            augmentations=resize,
        )

        model = Padim(
            backbone=args.backbone,
            layers=['layer1', 'layer2', 'layer3'],
        )

        export_root = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(export_root, exist_ok=True)

        engine = Engine(default_root_dir=tmpdir)

        print('Fitting PaDiM Gaussians (single pass, no backprop)...',
              flush=True)
        engine.fit(model=model, datamodule=datamodule)

        print('Evaluating on test set...', flush=True)
        engine.test(model=model, datamodule=datamodule)

        print('Exporting model...', flush=True)
        engine.export(
            model=model,
            export_type=ExportType.TORCH,
            export_root=export_root,
        )

        # anomalib writes to <export_root>/weights/torch/model.pt — move it
        candidate = os.path.join(export_root, 'weights', 'torch', 'model.pt')
        if os.path.exists(candidate):
            shutil.move(candidate, args.output)
            shutil.rmtree(os.path.join(export_root, 'weights'),
                          ignore_errors=True)

        print(f'Saved → {args.output}')

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
