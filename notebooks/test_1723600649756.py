import json

@hydra.main(version_base=None, config_path='../configs', config_name='experiment/example1')
def main(cfg: DictConfig):
    # 将DictConfig转换为字典
    cfg_dict = OmegaConf.to_container(cfg)
    # 打印JSON格式的配置
    print(json.dumps(cfg_dict, indent=2))
