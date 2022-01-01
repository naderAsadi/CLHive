import hydra
from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="defaults")
def parser(cfg: DictConfig) -> DictConfig:
    print(cfg)
    return cfg

# if __name__ == "__main__":
#     parser()