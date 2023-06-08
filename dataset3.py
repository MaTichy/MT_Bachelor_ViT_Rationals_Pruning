
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

class SVHNDataModule(LightningDataModule):

    def __init__(self, data_dir: str = "../data/SVHN", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224,224)), #32
            transforms.RandomCrop(224, padding=4), #32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        ])

    def prepare_data(self):
        # download only
        datasets.SVHN(self.data_dir, split='train', download=True)
        datasets.SVHN(self.data_dir, split='test', download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            svhn_full = datasets.SVHN(self.data_dir, split='train', transform=self.transform)
            self.svhn_train, self.svhn_val = random_split(svhn_full, [int(len(svhn_full)*0.8), len(svhn_full) - int(len(svhn_full)*0.8)])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.svhn_test = datasets.SVHN(self.data_dir, split='test', transform=self.test_transform)

    def train_dataloader(self):
        train_loader=DataLoader(self.svhn_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return train_loader

    def val_dataloader(self):
        val_loader=DataLoader(self.svhn_val, batch_size=self.batch_size, num_workers=8)
        return val_loader
    
    def test_dataloader(self):
        test_loader=DataLoader(self.svhn_test, batch_size=self.batch_size, num_workers=8)
        return test_loader
    

