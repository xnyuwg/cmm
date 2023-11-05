from cmm.conf.ESG_report_learning_conf import ESGReportLearningConfig
from cmm.data_processor.ESG_report_processor import ESGReportProcessor
from cmm.data_preparer.ESG_report_preparer import ESGReportPreparer
from cmm.metric.ESG_learning_metric import ESGLearningMetric
from cmm.model.ESG_learning_tree_split_model import ESGTreeModel
from cmm.optimizer.basic_optimizer import BasicOptimizer
from cmm.trainer.ESG_learning_tree_trainer import ESGLearningTreeTrainer
from cmm.conf.global_config_manager import GlobalConfigManager
from cmm.utils.util_structure import UtilStructure
import argparse
import torch
import importlib
import logging
importlib.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def get_config(args) -> ESGReportLearningConfig:
    if not torch.cuda.is_available():
        logging.error("cuda unavailable")
        exit()

    config = ESGReportLearningConfig()

    config.model_save_name = args.run_name
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.max_epochs = args.max_epochs
    config.use_partial_data = args.use_partial_data

    config.num_works = 0
    config.to_process_pipline = ('text', 'toc', 'size_stat', 'font_stat', 'tree_answer', )
    config.to_process_pipline = []
    config.data_loader_read_pipeline = ('text', 'toc', 'tree_answer')
    config.lm_model_name = 'roberta-base'
    config.max_node_per_batch = 256
    config.sentence_token_max_length = 256
    config.device = torch.device('cuda')
    config.hidden_sen_emb_dim = 128
    config.sentence_lm_embedding_dim = 768
    config.data_path = GlobalConfigManager.get_data_save_path()
    config.result_path = GlobalConfigManager.get_result_save_path()
    config.cache_path = GlobalConfigManager.get_cache_save_path()
    return config


def run(args):
    config = get_config(args)

    pro = ESGReportProcessor(config)

    pre = ESGReportPreparer(config=config, processor=pro)

    metric = ESGLearningMetric(config=config, preparer=pre)

    model = ESGTreeModel(config=config,
                         processor=pro,
                         )
    model.to(config.device)

    optimizer = BasicOptimizer(config=config,
                               model=model,
                               )

    trainer = ESGLearningTreeTrainer(config=config,
                                     model=model,
                                     optimizer=optimizer,
                                     preparer=pre,
                                     metric=metric,
                                     )
    trainer.train()


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--run_name", type=str, required=False, default='exp0', help="the name of this run")
    arg_parser.add_argument("--gradient_accumulation_steps", type=int, required=False, default=32, help="gradient_accumulation_steps")
    arg_parser.add_argument("--max_epochs", type=int, required=False, default=100, help="max_epochs")
    arg_parser.add_argument("--use_partial_data", type=str, required=False, default='False', help="whether to user partial version data")
    args = arg_parser.parse_args()
    args.use_partial_data = UtilStructure.str_to_bool(args.use_partial_data)
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)
