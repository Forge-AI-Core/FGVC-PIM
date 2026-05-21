import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def build_loader(args):
    train_set, train_loader = None, None
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        
        # Calculate weights for WeightedRandomSampler
        labels = [info["label"] for info in train_set.data_infos]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in labels]
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            num_workers=args.num_workers, 
            batch_size=args.batch_size, 
            sampler=sampler
        )

    val_set, val_loader = None, None
    if args.val_root is not None:
        val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index
        self.istrain = istrain

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        resize_size = int(data_size * 1.33)
        if istrain:
            # 소수 클래스(danger, excluded) 전용 추가 증강
            # RandomRotation: 스크랩은 무작위 방향으로 놓임
            # ColorJitter: 조명 변화 대응 (transforms_post보다 약하게)
            self.minority_augment = transforms.Compose([
                transforms.RandomRotation(degrees=45),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            ])
            # PIL 단계 transforms (RandAugment 적용 전까지)
            # RandomVerticalFlip 추가: 철스크랩은 어느 방향으로든 놓임
            self.transforms_pre = transforms.Compose([
                        transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                ])
            # Tensor 단계 transforms
            self.transforms_post = transforms.Compose([
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root):
        data_infos = []
        folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        valid_exts = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        for class_id, folder in enumerate(folders):
            files = [f for f in os.listdir(os.path.join(root, folder)) 
                     if not f.startswith('.') and f.lower().endswith(valid_exts)]
            for file in files:
                data_path = os.path.join(root, folder, file)
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        img = Image.fromarray(img)

        if self.istrain:
            # 1단계: resize → crop → flip (PIL 상태)
            img = self.transforms_pre(img)
            # 2단계: 소수 클래스(danger=1, excluded=2)에만 추가 증강 적용
            if label in [1, 2]:
                img = self.minority_augment(img)
            # 3단계: GaussianBlur, Sharpness, ToTensor, Normalize
            img = self.transforms_post(img)
        else:
            img = self.transforms(img)
        
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label
