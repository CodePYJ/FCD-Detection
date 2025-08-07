import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from modules.utils import instantiate_from_config
import torch

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=str,
        default="./configs/config.yaml"
    )

    parser.add_argument(
        "--test",
        type=bool,
        default=False
    )
    return parser


class DataModuleFromConfig:
    def __init__(self, batch_size, shuffle=False, train=None, test=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        if train is not None:
            self.train_config = train
            self.test_config = test
            self.train_dataloader = self._train_dataloader
            self.test_dataloader = self._test_dataloader

    def _train_dataloader(self):
        print("loading data...", flush=True)
        dataset = instantiate_from_config(self.train_config)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def _test_dataloader(self):
        print("loading data...", flush=True)
        dataset = instantiate_from_config(self.test_config)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = OmegaConf.load(args.base)
    model = instantiate_from_config(config.model)
    data = instantiate_from_config(config.data)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    if not args.test:
        dataLoader = data.train_dataloader()
        vali_dataLoader = data.test_dataloader()
        trainer = instantiate_from_config(config.trainer)
        trainer.setup(model, dataLoader, vali_dataLoader)
        trainer.train()
    else:
        dataLoader = data.test_dataloader()
        tester = instantiate_from_config(config.tester)
        tester.setup(model, dataLoader)
        tester.test()

