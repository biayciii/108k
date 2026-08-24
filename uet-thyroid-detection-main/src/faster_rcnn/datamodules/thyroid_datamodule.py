import os
from typing import Any, Dict, List, Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from . import collate_fn, get_test_transforms, get_train_transforms
from .components.thyroid_dataset import ThyroidDataset


class ThyroidDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        # train_val_split: List[float] = [0.9, 0.1],
        target_width: int = 256,
        target_height: int = 256,
        batch_size: int = 8,
        num_workers: int = 8,
        num_level: int = 5,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self._train_transforms = get_train_transforms()
        self._test_transforms = get_test_transforms()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.trainset = ThyroidDataset(
                os.path.join(self.hparams.data_dir, "images", "train"),
                os.path.join(self.hparams.data_dir, "annotations", "train.csv"),
                self.hparams.target_width,
                self.hparams.target_height,
                self.hparams.num_level,
                transforms=self._train_transforms,
            )

            self.valset = ThyroidDataset(
                os.path.join(self.hparams.data_dir, "images", "val"),
                os.path.join(self.hparams.data_dir, "annotations", "val.csv"),
                self.hparams.target_width,
                self.hparams.target_height,
                self.hparams.num_level,
                transforms=self._test_transforms,
            )

            self.testset = ThyroidDataset(
                os.path.join(self.hparams.data_dir, "images", "test"),
                os.path.join(self.hparams.data_dir, "annotations", "test.csv"),
                self.hparams.target_width,
                self.hparams.target_height,
                self.hparams.num_level,
                transforms=self._test_transforms,
            )

            # n_train = int(len(trainset)*self.hparams.train_val_split[0])
            # n_val =  len(trainset) - n_train

            # self.trainset, self.valset = random_split(
            #     dataset=trainset,
            #     lengths=[n_train, n_val],
            #     generator=torch.Generator().manual_seed(42),
            # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.testset,
            batch_size=self.hparams.batch_size,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
