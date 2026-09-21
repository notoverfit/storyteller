import json
import os

from src.configs.configs import Config

def write_config(config: Config, path: str, overwrite=False) -> None:
    config_json = json.dumps(config.__dict__)

    if not overwrite and os.path.exists(path):
        raise Exception('write_config :: path already exists and overwrite is false')

    with open(path, 'w') as f:
        f.write(config_json)
