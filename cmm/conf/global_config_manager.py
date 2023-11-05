import logging
import os
import configparser
from pathlib import Path


class GlobalConfigManager:
    """ init the config and current path """
    current_path = Path(os.path.split(os.path.realpath(__file__))[0] + '/../../')  # the path of the code .py file
    logging.info("Current Path: {}".format(current_path))
    config = configparser.ConfigParser()

    @classmethod
    def if_not_exist_then_creat(cls, path):
        if not os.path.exists(path):
            logging.info("Path not exist {}, creating...".format(path))
            os.makedirs(path)

    @classmethod
    def get_current_path(cls):
        return cls.current_path

    @classmethod
    def get_transformers_cache_path(cls):
        path = cls.current_path / 'cache' / 'transformer_cache'
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_model_save_path(cls):
        path = cls.current_path / 'cache' / 'model'
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_data_save_path(cls):
        path = cls.current_path / 'data'
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_result_save_path(cls):
        path = cls.current_path / 'cache' / 'result'
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_cache_save_path(cls):
        path = cls.current_path / 'cache'
        cls.if_not_exist_then_creat(path)
        return path
