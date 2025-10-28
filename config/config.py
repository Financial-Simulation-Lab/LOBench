"""
Created on : 2024-07-13
Created by : Mythezone
Updated by : Mythezone
Email      : mythezone@gmail.com
FileName   : ~/project/simlob-refined/config/config.py
Description: Configuration Class
---
Updated    : 
---
Todo       : 
"""

# Insert the path into sys.path for importing.
import sys,os,json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.single import SingletonMeta
from types import SimpleNamespace

class ConfigManager(metaclass=SingletonMeta):
    def __init__(self, config_dir='./',config_file='config.json'):
        self.config_dir=config_dir

        self.load_config(config_file)

    def load_config(self, config_file='config.json'):
        self.config_file = config_file
        config_path = os.path.join(self.config_dir, config_file)
        if os.path.isfile(config_path):
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
                self._update_attributes(config_data)
        else:
            raise FileNotFoundError(f"No {config_file} found in {self.config_dir}")

    def _update_attributes(self, config_data, prefix=''):
        for key, value in config_data.items():
            if isinstance(value, dict):
                self._update_attributes(value, prefix + key + '.')
            else:
                attribute_name = prefix + key
                self._set_nested_attribute(attribute_name, value)

    def _set_nested_attribute(self, attribute_name, value):
        parts = attribute_name.split('.')
        obj = self
        for part in parts[:-1]:
            if not hasattr(obj, part):
                setattr(obj, part, SimpleNamespace())
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def __getattr__(self, name):
        raise AttributeError(f"Attribute '{name}' not found in {self.config_dir} | {self.config_file}")
    
class ExpConfigManager(ConfigManager):
    def __init__(self, config_dir=None, config_file='default.json'):
        super().__init__(config_dir,config_file)
      
      
        
class Config:
    def __init__(self, config_file_path=None):
        self.config_file_path = config_file_path
        if self.config_file_path is not None:
            self.update_config(self.config_file_path)

    def update_config(self, config_file_path):
        self.config_file_path = config_file_path
        if os.path.isfile(config_file_path):
            with open(config_file_path, 'r') as config_file:
                config_data = json.load(config_file)
                self._update_attributes(config_data)
        else:
            raise FileNotFoundError(f"{config_file_path} is not file or can't be open.")

    def _update_attributes(self, config_data, prefix=''):
        for key, value in config_data.items():
            if isinstance(value, dict):
                self._update_attributes(value, prefix + key + '.')
            else:
                attribute_name = prefix + key
                self._set_nested_attribute(attribute_name, value)

    def _set_nested_attribute(self, attribute_name, value):
        parts = attribute_name.split('.')
        obj = self
        for part in parts[:-1]:
            if not hasattr(obj, part):
                setattr(obj, part, SimpleNamespace())
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def __getattr__(self, name):
        raise AttributeError(f"Attribute '{name}' not found in {self.config_file_path}")

    def save(self, save_path = None ):
        config_data = self._to_dict(self)
        if save_path is None:
            save_path = self.config_file_path 
        with open(save_path, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)

    def _to_dict(self, obj):
        if isinstance(obj, SimpleNamespace):
            return {k: self._to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, Config):
            return {k: self._to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        else:
            return obj

    def update_from_dict(self, update_dict):
        self._update_attributes(update_dict)
        
    def get_dict(self, key):
        parts = key.split('.')
        obj = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise AttributeError(f"Attribute '{key}' not found in {self.config_file_path}")
        return self._to_dict(obj)
        
    

# Usage example
if __name__ == "__main__":

    c = Config('./config.json')
    dct = {
        "updated": True,
        "folders":{
            "updated_folder": "updated_folder"
        }
    }
    c.update_config("experiment/exp_settings/default.json")
    # c.update_from_dict(dct)
    # c.folders.updated_folder = "updated_folder2"
    # # c.save("./update_config.json")
    # print(c.get_dict('folders'))
    c.save("./check_update.json")