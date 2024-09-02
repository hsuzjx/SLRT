import json

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='../configs', config_name='example1')
def main(cfg: DictConfig):
    # 将DictConfig转换为字典
    cfg_dict = OmegaConf.to_container(cfg)

    # 打印JSON格式的配置
    print(json.dumps(cfg_dict, indent=2))


if __name__ == '__main__':
    main()
