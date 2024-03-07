import argparse
import math
import os
import pathlib

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.models.utils import resize

from efficientvit.seg_model_zoo import create_seg_model


#local datasets
from datasets.rellis import RellisDataset


class Resize(object):
    def __init__(
        self,
        crop_size: tuple[int, int] or None,
        interpolation: int or None = cv2.INTER_CUBIC,
    ):
        self.crop_size = crop_size
        self.interpolation = interpolation

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.crop_size is None or self.interpolation is None:
            return feed_dict

        image, target = feed_dict["data"], feed_dict["label"]
        height, width = self.crop_size

        h, w, _ = image.shape
        if width != w or height != h:
            image = cv2.resize(
                image,
                dsize=(width, height),
                interpolation=self.interpolation,
            )
        return {
            "data": image,
            "label": target,
        }


class ToTensor(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, feed_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        image, mask = feed_dict["data"], feed_dict["label"]
        image = image.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
        mask = torch.as_tensor(mask, dtype=torch.int64)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        return {
            "data": image,
            "label": mask,
        }




class SegIOU:
    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = (outputs + 1) * (targets != self.ignore_index)
        targets = (targets + 1) * (targets != self.ignore_index)
        intersections = outputs * (outputs == targets)

        outputs = torch.histc(
            outputs,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        targets = torch.histc(
            targets,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        intersections = torch.histc(
            intersections,
            bins=self.num_classes,
            min=1,
            max=self.num_classes,
        )
        unions = outputs + targets - intersections

        return {
            "i": intersections,
            "u": unions,
        }


class CityscapesDataset(Dataset):
    classes = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )
    class_colors = (
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    )
    label_map = np.array(
        (
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            0,  # road 7
            1,  # sidewalk 8
            -1,
            -1,
            2,  # building 11
            3,  # wall 12
            4,  # fence 13
            -1,
            -1,
            -1,
            5,  # pole 17
            -1,
            6,  # traffic light 19
            7,  # traffic sign 20
            8,  # vegetation 21
            9,  # terrain 22
            10,  # sky 23
            11,  # person 24
            12,  # rider 25
            13,  # car 26
            14,  # truck 27
            15,  # bus 28
            -1,
            -1,
            16,  # train 31
            17,  # motorcycle 32
            18,  # bicycle 33
        )
    )

    def __init__(self, data_dir: str, crop_size: tuple[int, int] or None = None):
        super().__init__()

        # load samples
        samples = []
        print(data_dir)
        for dirpath, _, fnames in os.walk(data_dir):
            for fname in sorted(fnames):
                suffix = pathlib.Path(fname).suffix
                if suffix not in [".png"]:
                    continue
                image_path = os.path.join(dirpath, fname)
                mask_path = image_path.replace("/leftImg8bit/", "/gtFine/").replace(
                    "_leftImg8bit.", "_gtFine_labelIds."
                )
                if not mask_path.endswith(".png"):
                    mask_path = ".".join([*mask_path.split(".")[:-1], "png"])
                samples.append((image_path, mask_path))
        self.samples = samples

        # build transform
        self.transform = transforms.Compose(
            [
                Resize(crop_size),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, any]:
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = self.label_map[mask]

        feed_dict = {
            "data": image,
            "label": mask,
        }
        feed_dict = self.transform(feed_dict)
        return {
            "index": index,
            "image_path": image_path,
            "mask_path": mask_path,
            **feed_dict,
        }


def get_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    colors: tuple or list,
    opacity=0.5,
) -> np.ndarray:
    image_shape = image.shape[:2]
    mask_shape = mask.shape
    if image_shape != mask_shape:
        mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_mask = np.zeros_like(image, dtype=np.uint8)
    for k, color in enumerate(colors):
        seg_mask[mask == k, :] = color
    canvas = seg_mask * opacity + image * (1 - opacity)
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def main():
    parser = argparse.ArgumentParser()    
    
    parser.add_argument("--path", type=str, default="/scratch/apicker/cityscapes/leftImg8bit/val")
    
    #parser.add_argument("--path", type=str, default="/scratch/apicker/rellis3d/leftImg8bit/val") 
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "rellis"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=1)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.batch_size = args.batch_size * max(len(device_list), 1)

    if args.dataset == "cityscapes":
        dataset = CityscapesDataset(args.path, (args.crop_size, args.crop_size * 2))
        print(f'len of dataset: {len(dataset)}')
    elif args.dataset == "rellis":
        dataset = RellisDataset(args.path, crop_size=(args.crop_size, args.crop_size))
        #dataset = RellisDataset(args.path, crop_size=(1200, 1900))
        
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f'Length of dataloader: {len(data_loader)}')

    model = create_seg_model(args.model, args.dataset, weight_url=args.weight_url)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
    interaction = AverageMeter(is_distributed=False)
    union = AverageMeter(is_distributed=False)
    iou = SegIOU(len(dataset.classes))
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on {args.dataset}") as t:
            for feed_dict in data_loader:
                images, mask = feed_dict["data"].cuda(), feed_dict["label"].cuda()
                
                # compute output
                output = model(images)
                test = output
                # resize the output to match the shape of the mask
                if output.shape[-2:] != mask.shape[-2:]:
                    output = resize(output, size=mask.shape[-2:])
                upsampled = output
                output = torch.argmax(output, dim=1)
                stats = iou(output, mask)
                interaction.update(stats["i"])
                union.update(stats["u"])

                t.set_postfix(
                    {
                        "mIOU": (interaction.sum / union.sum).cpu().mean().item() * 100,
                        "image_size": list(images.shape[-2:]),
                    }
                )
                t.update()

                if args.save_path is not None:
                    with open(os.path.join(args.save_path, "summary.txt"), "a") as fout:
                        for i, (idx, image_path) in enumerate(zip(feed_dict["index"], feed_dict["image_path"])):
                            pred = output[i].cpu().numpy()
                            raw_image = np.array(Image.open(image_path).convert("RGB"))
                            canvas = get_canvas(raw_image, pred, dataset.class_colors)
                            canvas = Image.fromarray(canvas)
                            canvas.save(os.path.join(args.save_path, f"{idx}.png"))
                            fout.write(f"{idx}:\t{image_path}\n")

    print(f"mIoU = {(interaction.sum / union.sum).cpu().mean().item() * 100:.3f}")
    #print(f'Size of output from the model: {test.shape}')
 #  print(f'Size of upsampled: {upsampled.shape}')
    


if __name__ == "__main__":
    main()
