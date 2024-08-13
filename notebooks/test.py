import os

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../configs', config_name='experiment/example1')
def main(cfg: DictConfig):
    print(cfg.get('experiment').get('save_dir'))


if __name__ == '__main__':
    main()
