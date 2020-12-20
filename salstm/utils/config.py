"""
Adapted from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
"""
import sys
from importlib import import_module
from pathlib import Path

from addict import Dict


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as exp:
            ex = exp
        else:
            return value
        raise ex


class Config:
    """
    A facility for config files.
    The interface is the same as a dict object and also allows
    access config values as attributes.

    Example:
        >>> cfg = Config('/home/configs/test.py')
        >>> cfg.only_filename
        "/home/configs/test.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/configs/test.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    def __init__(self, filename=None):
        path_obj = Path(filename)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file <{filename}> not found.")

        if filename.endswith('.py'):
            module_name = path_obj.stem
            if '.' in module_name:
                raise ValueError('Dots are not allowed in config file path.')

            sys.path.insert(0, str(path_obj.resolve().parent))
            mod = import_module(module_name)
            sys.path.pop(0)

            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }

            super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
            super(Config, self).__setattr__('_filename', filename)
            if filename:
                with open(filename, 'r') as file:
                    super(Config, self).__setattr__('_text', file.read())
            else:
                super(Config, self).__setattr__('_text', '')
        else:
            raise IOError('Only .py config files are supported!')

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)
