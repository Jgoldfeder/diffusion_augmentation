""" Dataset Factory

Original Script by Ross Wightman

Greatly expanded by Judah Goldfeder
"""
import os
from typing import Optional
import torch
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, KMNIST, FashionMNIST,ImageFolder,FGVCAircraft,Caltech101,Food101,Flowers102,OxfordIIITPet,Country211,Caltech256,DTD,FER2013,STL10,SUN397,SVHN,USPS,Cityscapes,CelebA
import dogs
import cub
import torchvision

datasets ={    
    'inaturalist': 10000,
    'cifar100': 100,
    'cifar10': 10,
    'mnist': 10,
    'kmnist': 10,
    'fashion_mnist': 10,
    'qmnist': 10,
    'places365': 365,
    'aircraft':102,
    'food101': 101,
    'flowers102': 102,
    'country211': 211,
    'stl10': 10,
    'sun397': 397,
    'svhn':10,
    'pets': 37,
    'caltech256': 257,# off by 1 indexing
    'caltech101': 102, # off by one indexing
    'dtd': 47,
    'dogs': 120,
    'cub2011': 200,
    'stanford_cars': 196,
    'eurosat': 10,
}

#broken link: fer2013,Cityscapes,CelebA,

try:
    from torchvision.datasets import Places365
    has_places365 = True
except ImportError:
    has_places365 = False
try:
    from torchvision.datasets import INaturalist
    has_inaturalist = True
except ImportError:
    has_inaturalist = False
try:
    from torchvision.datasets import QMNIST
    has_qmnist = True
except ImportError:
    has_qmnist = False
try:
    from torchvision.datasets import ImageNet
    has_imagenet = True
except ImportError:
    has_imagenet = False
import torchvision
import torchvision.transforms as transforms
from timm.data.dataset import IterableImageDataset, ImageDataset

_TORCH_BASIC_DS = dict(
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    kmnist=KMNIST,
    fashion_mnist=FashionMNIST,
    usps = USPS,
)
_TRAIN_SYNONYM = dict(train=None, training=None)
_EVAL_SYNONYM = dict(val=None, valid=None, validation=None, eval=None, evaluation=None)




def create_dataset(
        name: str,
        root: Optional[str] = None,
        split: str = 'validation',
        class_map: dict = None,
        load_bytes: bool = False,
        download: bool = True,
        num_samples: Optional[int] = None,
        seed: int = 42,
        input_img_mode: str = 'RGB',
        is_training=True,
        batch_size=1,
        repeats=1,
        **kwargs,
):
    """ Dataset factory method

    Args:
        name: dataset name, empty is okay for folder based datasets
        root: root folder of dataset (all)
        split: dataset split (all)
            `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (folder, torch/folder)
        class_map: specify class -> index mapping via text file or dict (folder)
        load_bytes: load data, return images as undecoded bytes (folder)
        download: download dataset if not present and supported (HFDS, TFDS, torch)
            For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS, WDS)
        seed: seed for iterable datasets (TFDS, WDS)
        input_img_mode: Input image color conversion mode e.g. 'RGB', 'L' (folder, TFDS, WDS, HFDS)
        **kwargs: other args to pass to dataset

    Returns:
        Dataset object
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    name = name.lower()
    if name.startswith('torch/'):
        name = name.split('/', 2)[-1]
        torch_kwargs = dict(root=root, download=download, **kwargs)
        if name in _TORCH_BASIC_DS:
            ds_class = _TORCH_BASIC_DS[name]
            use_train = split in _TRAIN_SYNONYM
            ds = ds_class(train=use_train, **torch_kwargs,)
        elif name == 'inaturalist' or name == 'inat':
            assert has_inaturalist, 'Please update to PyTorch 1.10, torchvision 0.11+ for Inaturalist'
            target_type = 'full'
            split_split = split.split('/')
            if len(split_split) > 1:
                target_type = split_split[0].split('_')
                if len(target_type) == 1:
                    target_type = target_type[0]
                split = split_split[-1]
            if split in _TRAIN_SYNONYM:
                split = '2021_train'
            elif split in _EVAL_SYNONYM:
                split = '2021_valid'
            ds = INaturalist(version=split, target_type=target_type, **torch_kwargs)
        elif name == 'places365':
            assert has_places365, 'Please update to a newer PyTorch and torchvision for Places365 dataset.'
            if split in _TRAIN_SYNONYM:
                split = 'train-standard'
            elif split in _EVAL_SYNONYM:
                split = 'val'
            transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
            ds = Places365(split=split, **torch_kwargs,transform=transform)
        elif name == 'qmnist':
            assert has_qmnist, 'Please update to a newer PyTorch and torchvision for QMNIST dataset.'
            use_train = split in _TRAIN_SYNONYM
            ds = QMNIST(train=use_train, **torch_kwargs)
        elif name == 'imagenet':
            assert has_imagenet, 'Please update to a newer PyTorch and torchvision for ImageNet dataset.'
            if split in _EVAL_SYNONYM:
                split = 'val'
            ds = ImageNet(split=split, **torch_kwargs)
        elif name == 'aircraft':
            if split in _TRAIN_SYNONYM:
                split = 'trainval'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = FGVCAircraft(split=split, **torch_kwargs)
        elif name == 'food101':
            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = Food101(split=split, **torch_kwargs)
        elif name == 'flowers102':
            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = Flowers102(split=split, **torch_kwargs)
        elif name == 'fer2013':
            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = FER2013(split=split, **torch_kwargs)
        
        elif name == 'country211':
            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = Country211(split=split, **torch_kwargs)
        elif name == 'celeba':
            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = CelebA(split=split, **torch_kwargs)
        
        elif name == 'stl10':
            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = STL10(split=split, **torch_kwargs)

        elif name == 'cityscapes':
            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = Cityscapes(split=split, **torch_kwargs)

        
        elif name == 'sun397':
            transform = transforms.Compose([transforms.ToTensor()])

            generator = torch.Generator().manual_seed(42)
            full_dataset = SUN397(**torch_kwargs,transform=transform)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],generator=generator)
            if split in _TRAIN_SYNONYM:
                ds = Wrapper(train_dataset)
            elif split in _EVAL_SYNONYM:
                ds = Wrapper(test_dataset)       
        elif name == 'svhn':
            transform = transforms.Compose([transforms.ToTensor()])

            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = SVHN(split=split, **torch_kwargs,transform=transform)
        
        elif name == 'pets':
            transform = transforms.Compose([transforms.ToTensor()])

            if split in _TRAIN_SYNONYM:
                split = 'trainval'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = OxfordIIITPet(split=split, **torch_kwargs,transform=transform)

        
        elif name == 'caltech101':
            def f(x):
                if x.shape[0] == 1:
                    return x.repeat(3,1,1)
                return x
            transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: f(x))])
            generator = torch.Generator().manual_seed(42)
            full_dataset = Caltech101(**torch_kwargs,transform=transform)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],generator=generator)
            if split in _TRAIN_SYNONYM:
                ds = Wrapper(train_dataset)
            elif split in _EVAL_SYNONYM:
                ds = Wrapper(test_dataset)

        elif name == 'caltech256':
            import torchvision.transforms.functional as F

            def f(x):
                if not torch.is_tensor(x):
                    x=F.to_tensor(x)
                if x.shape[0] == 1:
                    return x.repeat(3,1,1)
                return F.to_pil_image(x)
            transform = transforms.Compose([transforms.Lambda(lambda x: f(x))])
            generator = torch.Generator().manual_seed(42)
            full_dataset = Caltech256(**torch_kwargs,transform=transform)
            print(root)
            #full_dataset = torchvision.datasets.ImageFolder(root=root+"/256_ObjectCategories/",transform=transform)
            #print(full_dataset.classes)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],generator=generator)
            if split in _TRAIN_SYNONYM:
                ds = Wrapper(train_dataset)
            elif split in _EVAL_SYNONYM:
                ds = Wrapper(test_dataset)
                
        elif name == 'cifar100-512':
            def f(x):
                if x.shape[0] == 1:
                    return x.repeat(3,1,1)
                return x
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: f(x))])
            generator = torch.Generator().manual_seed(42)
            # full_dataset = Caltech256(**torch_kwargs,transform=transform)
            print(root)
            # full_dataset = torchvision.datasets.ImageFolder(root=root+"/256_ObjectCategories/",transform=transform)
            from fewshot_dataset import ImageDatasetWithFilename
            full_dataset = ImageDatasetWithFilename(root, transform=transform)
            # print(full_dataset.classes)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],generator=generator)
            if split in _TRAIN_SYNONYM:
                ds = train_dataset
            elif split in _EVAL_SYNONYM:
                ds = test_dataset

                ds = Wrapper(test_dataset)

        # elif name == 'caltech256_o':
        #     def f(x):
        #         if x.shape[0] == 1:
        #             return x.repeat(3,1,1)
        #         return x
        #     transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Lambda(lambda x: f(x))])
        #     generator = torch.Generator().manual_seed(42)
        #     full_dataset = Caltech256(**torch_kwargs,transform=transform)
            
        #     print(torch_kwargs)
        #     #transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: f(x))])
        #     #print(root)
        #     #full_dataset = torchvision.datasets.ImageFolder(root=root+"/256_ObjectCategories/",transform=transform)
        #     #print(full_dataset.classes)
            
        #     train_size = int(0.8 * len(full_dataset))
        #     test_size = len(full_dataset) - train_size
        #     train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],generator=generator)
        #     if split in _TRAIN_SYNONYM:
        #         ds = train_dataset
        #     elif split in _EVAL_SYNONYM:
        #         ds = test_dataset
                
        
        elif name == 'dtd':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Lambda(lambda x: f(x))])

            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'val'
            ds = DTD(split=split, **torch_kwargs,transform=transform)
        elif name == 'dogs':
            input_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
    
            train_dataset = dogs.dogs(root=root,
                                     train=True,
                                     cropped=False,
                                     transform=input_transforms,
                                     download=True)
            test_dataset = dogs.dogs(root=root,
                                    train=False,
                                    cropped=False,
                                    transform=input_transforms,
                                    download=True)
            if split in _TRAIN_SYNONYM:
                ds = train_dataset
            elif split in _EVAL_SYNONYM:
                ds = test_dataset 

        elif name == 'cub2011':
            if split in _TRAIN_SYNONYM:
                ds = cub.Cub2011('./cub2011', train=True, download=True)
            elif split in _EVAL_SYNONYM:
                ds = cub.Cub2011('./cub2011', train=False, download=True)

        
        else:
            assert False, f"Unknown torchvision dataset {name}"
    elif name.startswith('hfds/'):
        # NOTE right now, HF datasets default arrow format is a random-access Dataset,
        # There will be a IterableDataset variant too, TBD
        if "stanford_cars" in name:
            if split in _TRAIN_SYNONYM:
                name = "hfds/Multimodal-Fatima/StanfordCars_train"
                split = "train"
            elif split in _EVAL_SYNONYM:
                name = "hfds/Multimodal-Fatima/StanfordCars_test"
                split = "test"
        if "eurosat" in name:
            if split in _TRAIN_SYNONYM:
                name = "hfds/timm/eurosat-rgb"
                split = "train"
            elif split in _EVAL_SYNONYM:
                name = "hfds/timm/eurosat-rgb"
                split = "test"
        # if "fer2013" in name:
        #     if split in _TRAIN_SYNONYM:
        #         name = "hfds/clip-benchmark/wds_fer2013"
        #         split = "train"
        #     elif split in _EVAL_SYNONYM:
        #         name = "hfds/clip-benchmark/wds_fer2013"
        #         split = "test"
        if "pets" in name:
                if split in _TRAIN_SYNONYM:
                    name = "hfds/timm/oxford-iiit-pet"
                    split = "train"
                elif split in _EVAL_SYNONYM:
                    name = "hfds/timm/oxford-iiit-pet"
                    split = "test"
        
        ds = ImageDataset(
            root,
            reader=name,
            split=split,
            class_map=class_map,
            input_img_mode=input_img_mode,
            **kwargs,
        )   
    return ds



from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
class SubClassDataSet(Dataset):
    def __init__(self, ds, classes):
        print(len(classes), " way classification")
        self.ds = ds
        self.classes = classes
        
        # try saving and loading
        fname = str(len(ds)) + "."
        for c in classes:
            fname+= str(c) + "."
        fname +="npy"
        if os.path.isfile(fname):
            self.indices = np.load(fname)
        else:        
            d_classes = {}
            self.indices=[]
            for c in classes:
                d_classes[c]=True
            classes=d_classes
            for x in range(len(ds)):
                _, target = ds[x]
                print(len(ds),x)
    
                if int(target) in classes:
                    self.indices.append(x)
            self.indices  = np.array(self.indices)
            np.save(fname,self.indices)
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        x,y = self.ds[idx]
        if torch.is_tensor(x):
            x = F.to_pil_image(x)

        return self.transform(x),y


from torch.utils.data import Dataset
class Wrapper(Dataset):
    def __init__(self, ds,transform=None):
        self.ds = ds
        self.transform=transform
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.ds.__getitem__(idx)
        img,lbl = self.ds.__getitem__(idx)
        if torch.is_tensor(img):
            img = F.to_pil_image(img)
        return self.transform(img),lbl