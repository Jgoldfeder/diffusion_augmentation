""" Dataset Factory

Original Script by Ross Wightman

Greatly expanded by Judah Goldfeder
"""
import os
import numpy as np
import torch
import dogs
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import Optional
from torchvision.datasets import FGVCAircraft, Food101, Flowers102, \
    OxfordIIITPet, Caltech256, SUN397
from timm.data.dataset import IterableImageDataset, ImageDataset
from torch.utils.data import Dataset

datasets ={    
    'aircraft':102,
    'food101': 101,
    'flowers102': 102,
    'sun397': 397,
    'pets': 37,
    'caltech256': 257,# off by 1 indexing
    'dogs': 120,
    'stanford_cars': 196,
}

_TRAIN_SYNONYM = { "train", "training" }
_EVAL_SYNONYM = { "val", "valid", "validation", "eval", "evaluation" }


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
     
        if name == 'aircraft':
            if split in _TRAIN_SYNONYM:
                split = 'trainval'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = FGVCAircraft(split=split, **torch_kwargs)
            classes = ds.classes

        elif name == 'food101':
            if split in _TRAIN_SYNONYM:
                split = 'train'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = Food101(split=split, **torch_kwargs)
            classes = ds.classes

        #elif name == 'flowers102':
        #    if split in _TRAIN_SYNONYM:
        #        split = 'train'
        #    elif split in _EVAL_SYNONYM:
        #        split = 'test'
        #    ds = Flowers102(split=split, **torch_kwargs)
        

        elif name == 'sun397':
            transform = transforms.Compose([transforms.ToTensor()])

            generator = torch.Generator().manual_seed(seed)
            full_dataset = SUN397(**torch_kwargs,transform=transform)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, 
                [train_size, test_size], 
                generator=generator
                )

            if split in _TRAIN_SYNONYM:
                ds = Wrapper(train_dataset)
            elif split in _EVAL_SYNONYM:
                ds = Wrapper(test_dataset) 

                # # HACK SINCE PAT AUGMENTED WRONG DS
                # full_dataset = Wrapper(train_dataset)
                # train_size = int(0.6 * len(full_dataset))
                # test_size = len(full_dataset) - train_size
                # train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],generator=generator)
                # ds = Wrapper(test_dataset)
            classes = None
        
        elif name == 'pets':
            #transform = transforms.Compose([transforms.ToTensor()])

            if split in _TRAIN_SYNONYM:
                split = 'trainval'
            elif split in _EVAL_SYNONYM:
                split = 'test'
            ds = OxfordIIITPet(split=split, **torch_kwargs,transform=None)
            ds = Wrapper(ds)
            classes = ds.ds.classes

        elif name == 'caltech256':

            def f(x):
                if not torch.is_tensor(x):
                    x=F.to_tensor(x)
                if x.shape[0] == 1:
                    return x.repeat(3,1,1)
                return F.to_pil_image(x)
            transform = transforms.Compose([transforms.Lambda(lambda x: f(x))])
            generator = torch.Generator().manual_seed(seed)
            full_dataset = Caltech256(**torch_kwargs,transform=transform)
            print(root)
            classes = [el.split('.')[1] for el in sorted(os.listdir(f"{ root }/caltech256/256_ObjectCategories"))]
            #full_dataset = torchvision.datasets.ImageFolder(root=root+"/256_ObjectCategories/",transform=transform)
            #print(full_dataset.classes)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, 
                [train_size, test_size], 
                generator=generator
                )
            if split in _TRAIN_SYNONYM:
                ds = Wrapper(train_dataset)
            elif split in _EVAL_SYNONYM:
                ds = Wrapper(test_dataset)
                
    
        elif name == 'dogs':
            # input_transforms = transforms.Compose([
            #     transforms.RandomResizedCrop(224, ),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor()])
            input_transforms=None
    
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
            classes = ds.classes

        
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
            classes = ["am general hummer suv 2000", "acura rl sedan 2012", "acura tl sedan 2012", "acura tl type-s 2008", "acura tsx sedan 2012", "acura integra type r 2001", "acura zdx hatchback 2012", "aston martin v8 vantage convertible 2012", "aston martin v8 vantage coupe 2012", "aston martin virage convertible 2012", "aston martin virage coupe 2012", "audi rs 4 convertible 2008", "audi a5 coupe 2012", "audi tts coupe 2012", "audi r8 coupe 2012", "audi v8 sedan 1994", "audi 100 sedan 1994", "audi 100 wagon 1994", "audi tt hatchback 2011", "audi s6 sedan 2011", "audi s5 convertible 2012", "audi s5 coupe 2012", "audi s4 sedan 2012", "audi s4 sedan 2007", "audi tt rs coupe 2012", "bmw activehybrid 5 sedan 2012", "bmw 1 series convertible 2012", "bmw 1 series coupe 2012", "bmw 3 series sedan 2012", "bmw 3 series wagon 2012", "bmw 6 series convertible 2007", "bmw x5 suv 2007", "bmw x6 suv 2012", "bmw m3 coupe 2012", "bmw m5 sedan 2010", "bmw m6 convertible 2010", "bmw x3 suv 2012", "bmw z4 convertible 2012", "bentley continental supersports conv. convertible 2012", "bentley arnage sedan 2009", "bentley mulsanne sedan 2011", "bentley continental gt coupe 2012", "bentley continental gt coupe 2007", "bentley continental flying spur sedan 2007", "bugatti veyron 16.4 convertible 2009", "bugatti veyron 16.4 coupe 2009", "buick regal gs 2012", "buick rainier suv 2007", "buick verano sedan 2012", "buick enclave suv 2012", "cadillac cts-v sedan 2012", "cadillac srx suv 2012", "cadillac escalade ext crew cab 2007", "chevrolet silverado 1500 hybrid crew cab 2012", "chevrolet corvette convertible 2012", "chevrolet corvette zr1 2012", "chevrolet corvette ron fellows edition z06 2007", "chevrolet traverse suv 2012", "chevrolet camaro convertible 2012", "chevrolet hhr ss 2010", "chevrolet impala sedan 2007", "chevrolet tahoe hybrid suv 2012", "chevrolet sonic sedan 2012", "chevrolet express cargo van 2007", "chevrolet avalanche crew cab 2012", "chevrolet cobalt ss 2010", "chevrolet malibu hybrid sedan 2010", "chevrolet trailblazer ss 2009", "chevrolet silverado 2500hd regular cab 2012", "chevrolet silverado 1500 classic extended cab 2007", "chevrolet express van 2007", "chevrolet monte carlo coupe 2007", "chevrolet malibu sedan 2007", "chevrolet silverado 1500 extended cab 2012", "chevrolet silverado 1500 regular cab 2012", "chrysler aspen suv 2009", "chrysler sebring convertible 2010", "chrysler town and country minivan 2012", "chrysler 300 srt-8 2010", "chrysler crossfire convertible 2008", "chrysler pt cruiser convertible 2008", "daewoo nubira wagon 2002", "dodge caliber wagon 2012", "dodge caliber wagon 2007", "dodge caravan minivan 1997", "dodge ram pickup 3500 crew cab 2010", "dodge ram pickup 3500 quad cab 2009", "dodge sprinter cargo van 2009", "dodge journey suv 2012", "dodge dakota crew cab 2010", "dodge dakota club cab 2007", "dodge magnum wagon 2008", "dodge challenger srt8 2011", "dodge durango suv 2012", "dodge durango suv 2007", "dodge charger sedan 2012", "dodge charger srt-8 2009", "eagle talon hatchback 1998", "fiat 500 abarth 2012", "fiat 500 convertible 2012", "ferrari ff coupe 2012", "ferrari california convertible 2012", "ferrari 458 italia convertible 2012", "ferrari 458 italia coupe 2012", "fisker karma sedan 2012", "ford f-450 super duty crew cab 2012", "ford mustang convertible 2007", "ford freestar minivan 2007", "ford expedition el suv 2009", "ford edge suv 2012", "ford ranger supercab 2011", "ford gt coupe 2006", "ford f-150 regular cab 2012", "ford f-150 regular cab 2007", "ford focus sedan 2007", "ford e-series wagon van 2012", "ford fiesta sedan 2012", "gmc terrain suv 2012", "gmc savana van 2012", "gmc yukon hybrid suv 2012", "gmc acadia suv 2012", "gmc canyon extended cab 2012", "geo metro convertible 1993", "hummer h3t crew cab 2010", "hummer h2 sut crew cab 2009", "honda odyssey minivan 2012", "honda odyssey minivan 2007", "honda accord coupe 2012", "honda accord sedan 2012", "hyundai veloster hatchback 2012", "hyundai santa fe suv 2012", "hyundai tucson suv 2012", "hyundai veracruz suv 2012", "hyundai sonata hybrid sedan 2012", "hyundai elantra sedan 2007", "hyundai accent sedan 2012", "hyundai genesis sedan 2012", "hyundai sonata sedan 2012", "hyundai elantra touring hatchback 2012", "hyundai azera sedan 2012", "infiniti g coupe ipl 2012", "infiniti qx56 suv 2011", "isuzu ascender suv 2008", "jaguar xk xkr 2012", "jeep patriot suv 2012", "jeep wrangler suv 2012", "jeep liberty suv 2012", "jeep grand cherokee suv 2012", "jeep compass suv 2012", "lamborghini reventon coupe 2008", "lamborghini aventador coupe 2012", "lamborghini gallardo lp 570-4 superleggera 2012", "lamborghini diablo coupe 2001", "land rover range rover suv 2012", "land rover lr2 suv 2012", "lincoln town car sedan 2011", "mini cooper roadster convertible 2012", "maybach landaulet convertible 2012", "mazda tribute suv 2011", "mclaren mp4-12c coupe 2012", "mercedes-benz 300-class convertible 1993", "mercedes-benz c-class sedan 2012", "mercedes-benz sl-class coupe 2009", "mercedes-benz e-class sedan 2012", "mercedes-benz s-class sedan 2012", "mercedes-benz sprinter van 2012", "mitsubishi lancer sedan 2012", "nissan leaf hatchback 2012", "nissan nv passenger van 2012", "nissan juke hatchback 2012", "nissan 240sx coupe 1998", "plymouth neon coupe 1999", "porsche panamera sedan 2012", "ram c/v cargo van minivan 2012", "rolls-royce phantom drophead coupe convertible 2012", "rolls-royce ghost sedan 2012", "rolls-royce phantom sedan 2012", "scion xd hatchback 2012", "spyker c8 convertible 2009", "spyker c8 coupe 2009", "suzuki aerio sedan 2007", "suzuki kizashi sedan 2012", "suzuki sx4 hatchback 2012", "suzuki sx4 sedan 2012", "tesla model s sedan 2012", "toyota sequoia suv 2012", "toyota camry sedan 2012", "toyota corolla sedan 2012", "toyota 4runner suv 2012", "volkswagen golf hatchback 2012", "volkswagen golf hatchback 1991", "volkswagen beetle hatchback 2012", "volvo c30 hatchback 2012", "volvo 240 sedan 1993", "volvo xc90 suv 2007", "smart fortwo convertible 2012"]

 
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
    return ds, classes


class SubClassDataSet(Dataset):
    def __init__(self, ds, classes):
        print(len(classes), " way classification")
        self.ds = ds
        self.classes = classes
        
        # try saving and loading
        fname = str(len(ds)) + "."
        for c in classes:
            fname+= str(c) + "."
        fname += "npy"
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
                print(len(ds), x)
    
                if int(target) in classes:
                    self.indices.append(x)
            self.indices  = np.array(self.indices)
            np.save(fname,self.indices)
        self.indices = self.indices.tolist()
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        x,y = self.ds[idx]
        if torch.is_tensor(x):
            x = F.to_pil_image(x)

        return self.transform(x),y


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