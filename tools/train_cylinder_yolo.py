#!/usr/bin/env python3
"""Train the custom cylinder YOLO segmentation model.

Example:
python3 tools/train_cylinder_yolo.py \
  --data ~/ris/ros_ws/src/rins_robot/config/cylinder_dataset.yaml \
  --model yolo11s-seg.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 8 \
  --device 0
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to Ultralytics dataset YAML')
    parser.add_argument('--model', default='yolo11s-seg.pt', help='Base segmentation model; yolo11s-seg is stronger than nano. Try yolo11m-seg for more accuracy if GPU/FPS allow it.')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='', help="'', 'cpu', '0', ...")
    parser.add_argument('--project', default='runs/cylinder_yolo')
    parser.add_argument('--name', default='cylinder_seg')
    parser.add_argument('--patience', type=int, default=25)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=(args.device if args.device else None),
        project=args.project,
        name=args.name,
        patience=args.patience,
        task='segment',
    )
    print(results)
    print('\nTraining finished.')
    print(f'Best model should be under: {args.project}/{args.name}/weights/best.pt')


if __name__ == '__main__':
    main()
