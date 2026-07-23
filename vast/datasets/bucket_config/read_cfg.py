import types
import os

class ConfigDict(dict):
    def get(self, key, default=None):
        return super().get(key, default)
    


def get_cfg(filename):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, filename)
    
    config_module = types.ModuleType("config")
    with open(full_path, "r") as f:
        exec(f.read(), config_module.__dict__)
    
    return ConfigDict({k: v for k, v in config_module.__dict__.items() if not k.startswith("__")})

# pass
# dic = get_cfg('stage3.py')
# print(dic)