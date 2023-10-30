from ruamel.yaml import YAML
import logging

class YParams():
  """ Yaml file parser """
  def __init__(self, yaml_filename, config_name, print_params=False):
    self._yaml_filename = yaml_filename
    self._config_name = config_name
    self.params = {}

    with open(yaml_filename) as _file:

      for key, val in YAML().load(_file)[config_name].items():
        if val =='None': val = None

        self.params[key] = val
        self.__setattr__(key, val)

    if print_params:
      self.log()

  def __getitem__(self, key):
    return self.params[key]

  def __setitem__(self, key, val):
    self.params[key] = val

  def get(self, key, default=None):
    """Get a parameter value"""
    if hasattr(self, key):
      return getattr(self, key)
    else:
      return self.params.get(key, default)

  def log(self):
    logging.info("------------------ Configuration ------------------")
    logging.info("Configuration file: "+str(self._yaml_filename))
    logging.info("Configuration name: "+str(self._config_name))
    for key, val in self.params.items():
      logging.info(str(key) + ' ' + str(val))
    logging.info("---------------------------------------------------")

  def update(self, new_params):
    self.params.update(new_params)
    for key, val in new_params.items():
      self.__setattr__(key, val)
