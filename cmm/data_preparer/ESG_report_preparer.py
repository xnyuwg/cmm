from cmm.data_processor.ESG_report_processor import ESGReportProcessor
from cmm.conf.ESG_report_learning_conf import ESGReportLearningConfig
from torch.utils.data import Dataset, DataLoader
import logging


class ESGDataSet(Dataset):
    @staticmethod
    def my_collate_fn(batch):
        assert len(batch) == 1
        batch = batch[0]
        read_pipeline = batch['read_pipline']
        record_res = {}
        tensor_res = {}
        record_res['to_name'] = batch['to_name']
        record_res['report_id'] = batch['report_id']
        record_res['tree_answer'] = batch['tree_answer']
        if 'text' or read_pipeline or 'toc' or read_pipeline:
            tensor_res['text_example'] = batch['text_example']
            tensor_res['toc_example'] = batch['toc_example']
        return record_res, tensor_res

    def __init__(self,
                 to_name_list,
                 get_data_fn,
                 read_pipeline,
                 start_index,
                 ):
        self.to_name_list = to_name_list
        self.get_data_fn = get_data_fn
        self.start_index = start_index
        self.read_pipeline = read_pipeline

    def __len__(self):
        return len(self.to_name_list)

    def __getitem__(self, index):
        to_name = self.to_name_list[index]
        data = self.get_data_fn(to_name=to_name)
        res = {}
        res['read_pipline'] = self.read_pipeline
        report_id = index + self.start_index
        res['report_id'] = report_id
        res['to_name'] = to_name
        if 'text' in self.read_pipeline or 'toc' in self.read_pipeline:
            res['text_example'] = data['text']
            res['toc_example'] = data['toc']
        if 'tree_answer' in self.read_pipeline:
            res['tree_answer'] = data['tree_answer']
        return res


class ESGReportPreparer:
    def __init__(self,
                 config: ESGReportLearningConfig,
                 processor: ESGReportProcessor,
                 ):
        self.lm_model_name = config.lm_model_name
        self.config = config
        to_name_list, get_data_fn = processor.get_fn_of_get_data(read_data=config.data_loader_read_pipeline)
        self.to_name_list = to_name_list
        self.get_data_fn = get_data_fn

        self.train_set = processor.train_set
        self.dev_set = processor.dev_set
        self.test_set = processor.test_set
        self.train_len = len(self.train_set)
        self.dev_len = len(self.dev_set)
        self.test_len = len(self.test_set)

    def get_fn_of_get_loader(self):
        def get_loader():
            train_dataset = ESGDataSet(self.train_set, self.get_data_fn, self.config.data_loader_read_pipeline, start_index=0)
            test_dataset = ESGDataSet(self.test_set, self.get_data_fn, self.config.data_loader_read_pipeline, start_index=self.train_len)
            dev_dataset = ESGDataSet(self.dev_set, self.get_data_fn, self.config.data_loader_read_pipeline, start_index=self.train_len + self.test_len)
            logging.info("dataset {} train, {} dev, {} test".format(len(train_dataset), len(dev_dataset), len(test_dataset)))

            logging.info("init data loader...")
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, collate_fn=ESGDataSet.my_collate_fn, shuffle=self.config.data_loader_shuffle, num_workers=self.config.num_works)
            dev_loader = DataLoader(dev_dataset, batch_size=self.config.batch_size, collate_fn=ESGDataSet.my_collate_fn, shuffle=False, num_workers=self.config.num_works)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, collate_fn=ESGDataSet.my_collate_fn, shuffle=False, num_workers=self.config.num_works)

            return train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader
        return get_loader
