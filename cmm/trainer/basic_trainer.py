import torch
import torch.utils.data
from cmm.model.basic_model import BasicModel
from cmm.conf.global_config_manager import GlobalConfigManager
from cmm.optimizer.basic_optimizer import BasicOptimizer
import json
import os
import logging
from torch.utils.data import DataLoader
from typing import List, Callable
from cmm.conf.basic_conf import BasicConfig


class BasicTrainer:
    @staticmethod
    def write_json_file(file_name, data):
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def result_folder_init(folder_name):
        path = GlobalConfigManager.get_result_save_path() / folder_name
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def epoch_format(epoch, length):
        epoch_formatted = str(epoch)
        epoch_formatted = '0' * (length - len(epoch_formatted)) + epoch_formatted
        return epoch_formatted

    def __init__(self,
                 config: BasicConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 train_loader: torch.utils.data.DataLoader,
                 dev_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 ):
        self.config = config
        self.device = config.device
        self.skip_save_train_score = False
        self.model: BasicModel = model
        self.optimizer: BasicOptimizer = optimizer
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.dev_loader: torch.utils.data.DataLoader = dev_loader
        self.test_loader: torch.utils.data.DataLoader = test_loader
        num_training_steps = (len(train_loader) * config.max_epochs) // config.gradient_accumulation_steps + 1
        self.optimizer.prepare_for_train(num_training_steps=num_training_steps,
                                         gradient_accumulation_steps=config.gradient_accumulation_steps)
        if config.model_load_name is not None:
            self.optimizer.load_model(GlobalConfigManager.get_model_save_path() / config.model_load_name)
        self.result_folder_path = self.result_folder_init(config.model_save_name)

    def basic_train_template(self,
                             train_batch_fn: Callable,
                             train_args: dict,
                             eval_batch_fn: Callable,
                             eval_args: dict,
                             test_batch_fn: Callable = None,
                             test_args: dict = None,
                             train_loader: DataLoader = None,
                             dev_loader: DataLoader = None,
                             test_loader: DataLoader = None,
                             ):
        test_batch_fn = eval_batch_fn if test_batch_fn is None else test_batch_fn
        test_args = eval_args if test_args is None else test_args
        train_loader = self.train_loader if train_loader is None else train_loader
        dev_loader = self.dev_loader if dev_loader is None else dev_loader
        test_loader = self.test_loader if test_loader is None else test_loader
        current_epoch = 0
        for epoch in range(current_epoch, self.config.max_epochs):
            train_batch_fn(dataloader=train_loader, epoch=epoch, **train_args)
            # uncomment to save model
            # self.optimizer.save_model(GlobalConfigManager.get_model_save_path() / (self.config.model_save_name + '_temp.pth'))
            logging.info('Eval Epoch = {}, dev:'.format(epoch))
            dev_score_results = eval_batch_fn(dataloader=dev_loader, epoch=epoch, **eval_args)
            logging.info('Eval Epoch = {}, test:'.format(epoch))
            test_score_results = test_batch_fn(dataloader=test_loader, epoch=epoch, **test_args)
            final_score_results = {'dev': dev_score_results,
                                   'test': test_score_results,
                                   "epoch": epoch,
                                   }
            epoch_formatted = self.epoch_format(epoch, 4)
            score_results_file_name = self.config.model_save_name + '_' + epoch_formatted + '.json'
            self.write_json_file(self.result_folder_path / score_results_file_name, final_score_results)
            logging.info('score result save to {}'.format(self.result_folder_path / score_results_file_name))
        logging.info("Train over!")
