import argparse

from omegaconf import OmegaConf

from config import ObjectsParamsGroups, ParamsGroups
from training import ObjectTrainer
from training import SceneTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object", action="store_true", help="generate object or scene"
    )
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    if args.object:
        pg = OmegaConf.structured(ObjectsParamsGroups)
        cfg = OmegaConf.merge(
            pg, OmegaConf.load(args.config), OmegaConf.from_cli(extras)
        )
        trainer = ObjectTrainer(cfg)
    else:
        pg = OmegaConf.structured(ParamsGroups())
        cfg = OmegaConf.merge(
            pg, OmegaConf.load(args.config), OmegaConf.from_cli(extras)
        )
        trainer = SceneTrainer(cfg)

    trainer.train()
