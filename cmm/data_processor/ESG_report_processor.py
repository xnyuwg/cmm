import logging
from typing import List, Dict
from cmm.conf.PDF_parser_conf import PDFParserConfig
from cmm.conf.ESG_report_learning_conf import ESGReportLearningConfig
from cmm.data_example.PDF_example import PDFBlockExample
from cmm.text_processore.pdf_parser import PDFParser
from tqdm import tqdm
from cmm.utils.util_data import UtilData
from cmm.utils.util_structure import UtilStructure
from anytree import Node
from anytree.exporter import JsonExporter
from cmm.utils.tree_edit_distance import TreeEditDistanceManager
from cmm.conf.global_config_manager import GlobalConfigManager
from cmm.data_example.document_info_example import DocumentInfoExample


class ESGReportProcessor:
    def __init__(self,
                 config: ESGReportLearningConfig,
                 ):
        self.lm_model_name = config.lm_model_name
        self.config = config
        self.report_summary_file = 'document_info.jsonl'
        self.text_folder = 'report_full'
        self.toc_folder = 'toc'
        self.processed_text_folder = 'processed_text'
        self.processed_toc_folder = 'processed_toc'
        self.processed_tree_answer_folder = 'ans'

        for to_creat_path in [self.processed_text_folder, self.processed_toc_folder, self.processed_tree_answer_folder]:
            GlobalConfigManager.if_not_exist_then_creat(self.config.cache_path / to_creat_path)

        pdf_parse_conf = self.get_pdf_parser_config()
        self.report_parser = PDFParser(config=pdf_parse_conf)

        self.label_ratio = None

        self.init()

    def init(self):
        self.summary_examples = self.read_document_info_as_examples()

        if self.config.use_partial_data:
            self.train_set = [x.doc_id for x in self.summary_examples if x.in_partial and x.split_partial == 'train']
            self.dev_set = [x.doc_id for x in self.summary_examples if x.in_partial and x.split_partial == 'dev']
            self.test_set = [x.doc_id for x in self.summary_examples if x.in_partial and x.split_partial == 'test']
        else:
            self.train_set = [x.doc_id for x in self.summary_examples if x.split_full == 'train']
            self.dev_set = [x.doc_id for x in self.summary_examples if x.split_full == 'dev']
            self.test_set = [x.doc_id for x in self.summary_examples if x.split_full == 'test']

        if 'text' in self.config.to_process_pipline or 'toc' in self.config.to_process_pipline:
            logging.info('text_process')
            self.report_text_toc_process(summary_examples=self.summary_examples,
                                         read_text_folder=self.text_folder,
                                         read_toc_folder=self.toc_folder,
                                         write_text_folder=self.processed_text_folder,
                                         write_toc_folder=self.processed_toc_folder,
                                         )

        if 'size_stat' in self.config.to_process_pipline:
            logging.info('size_stat_process')
            self.size_stat_process(summary_examples=self.summary_examples,
                                   read_text_folder=self.processed_text_folder,
                                   read_toc_folder=self.processed_toc_folder,
                                   )

        size_path = self.config.cache_path / 'size_stat.json'
        size_c = UtilData.read_raw_json_file(size_path)
        self.config.common_size_chose = size_c['common_size_chose']
        logging.info("common_size_chose={}".format(self.config.common_size_chose))

        if 'font_stat' in self.config.to_process_pipline:
            logging.info('font_stat_process')
            self.font_stat_process(summary_examples=self.summary_examples,
                                   read_text_folder=self.processed_text_folder,
                                   read_toc_folder=self.processed_toc_folder,
                                   )

        font_path = self.config.cache_path / 'font_stat.json'
        font_js = UtilData.read_raw_json_file(font_path)
        self.font_list = font_js['font_list']
        logging.info("font_list_len={}".format(len(self.font_list)))

        if 'tree_answer' in self.config.to_process_pipline:
            logging.info('processed_tree_answer')
            self.process_template(process_fn=self.tree_answer_process,
                                  summary_examples=self.summary_examples,
                                  read_text_folder=self.processed_text_folder,
                                  read_toc_folder=self.processed_toc_folder,
                                  write_folder=self.processed_tree_answer_folder,
                                  )

    def read_document_info_as_examples(self):
        summary_jsonl = UtilData.read_raw_jsonl_file(self.config.data_path / self.report_summary_file)
        summary_examples = [DocumentInfoExample().from_json(x) for x in summary_jsonl]
        return summary_examples

    def get_pdf_parser_config(self) -> PDFParserConfig:
        config = PDFParserConfig()
        return config

    def get_ids_from_texts(self, texts):
        ids_to_block = {}
        for page in texts:
            for block in page:
                ids_to_block[block.id] = block
        return ids_to_block

    def get_common_size_of_report(self, texts, common_size_chose):
        size_count = {}
        for papge in texts:
            for block in papge:
                UtilStructure.count_dict_update(size_count, block.size, len(block.text.split(' ')))
        size_count = UtilStructure.get_count_dict_sorted(size_count, reverse=True)
        the_size = None
        step = -1
        for sc in [x[0] for x in size_count]:
            if the_size is None:
                the_size = sc
                step += 1
            elif sc < the_size:
                the_size = sc
                step += 1
            if step == common_size_chose:
                break
        return the_size

    def capital_ratio(self, s):
        total_letters = sum(c.isalpha() for c in s)
        uppercase_letters = sum(1 for c in s if c.isupper())
        # If there are no letters in the string, return 0
        if total_letters == 0:
            return 0
        return uppercase_letters / total_letters

    def report_text_toc_process(self, summary_examples, read_text_folder, read_toc_folder, write_text_folder, write_toc_folder):
        for report in tqdm(summary_examples):
            text_path = self.config.data_path / read_text_folder / (report.doc_id + '.jsonl')
            text_example = UtilData.read_raw_jsonl_file(text_path, verbose=False)
            toc_path = self.config.data_path / read_toc_folder / (report.doc_id + '.jsonl')
            toc_example = UtilData.read_raw_jsonl_file(toc_path, verbose=False)
            # text
            first_page = True
            new_text = []
            block_count = 0
            root_block = {
                'id': 'r0',
                'text': 'root',
                'original_text': ['root'],
                'size': 100,
                'font': '',
                'color': 0,
                'position': [0, 1, 0, 1],
                'page': -1,
                'xy_cut_sequence_number': -1,
                'page_height': 1,
                'page_width': 1,
                'bboxes': [[0, 1, 0, 1]]
            }
            for page in text_example:
                new_page = []
                if first_page:
                    new_page.append(root_block)
                    block_count += 1
                    first_page = False
                for block in page:
                    new_page.append(block)
                    block_count += 1
                new_text.append(new_page)
            assert len(text_example) == len(new_text)
            new_text = self.size_aut(new_text)
            # toc
            root_toc = {
                'level': 0,
                'heading': 'root',
                'page': 0,
                'block_id': 'r0'
            }
            new_toc = [root_toc] + toc_example
            # write
            save_name = report.doc_id + '.jsonl'
            text_path = self.config.cache_path / write_text_folder / save_name
            toc_path = self.config.cache_path / write_toc_folder / save_name
            UtilData.write_jsonl_file_line_error_catching(text_path, new_text, default_line={}, verbose=False)
            UtilData.write_jsonl_file_line_error_catching(toc_path, new_toc, default_line=None, verbose=False)

    def size_aut(self, text_example):
        def text_fea_key(block):
            return str(block['font']) + ';' + str(block['color'])
        font_count_dic = {}
        for page in text_example:
            for block in page:
                UtilStructure.count_dict_update(font_count_dic, text_fea_key(block), len(block['text'].split(' ')))
        font_counts = [(k, v) for k, v in font_count_dic.items()]
        font_counts = sorted(font_counts, key=lambda x: x[1])
        font_shift = {fc[0]: i * 0.0001 for i, fc in enumerate(font_counts)}
        for page in text_example:
            for block in page:
                block['size'] = block['size'] - font_shift[text_fea_key(block)]
                if self.capital_ratio(block['text']) > 0.9:
                    block['size'] += 0.01
        return text_example

    def size_stat_process(self, summary_examples, read_text_folder, read_toc_folder):
        common_size_chose = -1
        toc_in_per = 0
        while toc_in_per < 0.95:
            common_size_chose += 1
            logging.info('size_stat_process on common_size_chose={} and lst toc_in_per={}'.format(common_size_chose, toc_in_per))
            total_toc = 0
            in_toc = 0
            # for report in tqdm(summary_examples):
            #     if report.doc_id not in self.train_set:
            #         continue
            for doc_id in self.train_set:
                path = self.config.cache_path / read_text_folder / (doc_id + '.jsonl')
                text_jsonl = UtilData.read_raw_jsonl_file(path, verbose=False)
                toc_path = self.config.cache_path / read_toc_folder / (doc_id + '.jsonl')
                toc_jsonl = UtilData.read_raw_jsonl_file(toc_path, verbose=False)
                text_example, toc_example = self.report_parser.get_report_toc_as_example(report=text_jsonl, toc=toc_jsonl)

                ids_to_block = self.get_ids_from_texts(text_example)
                the_size = self.get_common_size_of_report(text_example, common_size_chose)
                for toc in toc_example:
                    total_toc += 1
                    if ids_to_block[toc[3]].size > the_size:
                        in_toc += 1
            toc_in_per = in_toc / total_toc

        logging.info("common_size_chose={}, with toc_in_per={}".format(common_size_chose, toc_in_per))
        res = {
            'common_size_chose': common_size_chose,
            'toc_in_per': toc_in_per,
        }
        path = self.config.cache_path / 'size_stat.json'
        UtilData.write_json_file(path, res, verbose=False)

    def font_stat_process(self, summary_examples, read_text_folder, read_toc_folder):
        font_set = set()
        for report in tqdm(summary_examples):
            path = self.config.cache_path / read_text_folder / (report.doc_id + '.jsonl')
            text_jsonl = UtilData.read_raw_jsonl_file(path, verbose=False)
            text_example, toc_example = self.report_parser.get_report_toc_as_example(report=text_jsonl, toc=None)
            for page in text_example:
                for block in page:
                    font_set.add(block.font.lower())
        # logging.info("font_set_len={}, font_set_len={}".format(len(font_set), font_set))
        res = {
            'font_list': list(font_set),
        }
        path = self.config.cache_path / 'font_stat.json'
        UtilData.write_json_file(path, res, verbose=False)

    def process_template(self, process_fn, summary_examples, read_text_folder, read_toc_folder, write_folder=None):
        for report in tqdm(summary_examples):
            path = self.config.cache_path / read_text_folder / (report.doc_id + '.jsonl')
            text_jsonl = UtilData.read_raw_jsonl_file(path, verbose=False)
            toc_path = self.config.cache_path / read_toc_folder / (report.doc_id + '.jsonl')
            toc_jsonl = UtilData.read_raw_jsonl_file(toc_path, verbose=False)
            text_example, toc_example = self.report_parser.get_report_toc_as_example(report=text_jsonl, toc=toc_jsonl)
            if write_folder is None:
                process_fn(report.doc_id, text_example, toc_example)
            else:
                process_fn(report.doc_id, text_example, toc_example, write_folder)

    def tree_answer_process(self, to_name, text_example: List[List[PDFBlockExample]], toc_example: List[list], write_folder):
        ids_to_count = {}
        ids_to_block = {}
        count = -1
        for page in text_example:
            for block in page:
                count += 1
                ids_to_count[block.id] = count
                ids_to_block[block.id] = block

        node_list = []
        for toc in toc_example:
            node_list.append(Node(toc[3]))

        for i, toc in enumerate(toc_example):
            toc_node = node_list[i]
            for j in reversed(range(i)):
                before_toc = toc_example[j]
                before_node = node_list[j]
                if before_toc[0] < toc[0]:
                    toc_node.parent = before_node
                    break

        for node in node_list:
            children = node.children
            children = sorted(children, key=lambda x: int(ids_to_count[x.name]))
            node.children = children

        root_node = node_list[0]
        tree_id_str = TreeEditDistanceManager.convert_tree_to_string(root_node, value_fn=lambda x: x.name, children_fn=lambda x: x.children)
        tree_count_str = TreeEditDistanceManager.convert_tree_to_string(root_node, value_fn=lambda x: str(ids_to_count[x.name]), children_fn=lambda x: x.children)
        tree_text_str = TreeEditDistanceManager.convert_tree_to_string(root_node, value_fn=lambda x: str(ids_to_block[x.name].text), children_fn=lambda x: x.children)

        exporter = JsonExporter(indent=4, ensure_ascii=False)

        res = {
            'tree_id_str': tree_id_str,
            'tree_count_str': tree_count_str,
            'tree_text_str': tree_text_str,
            'tree_node_list': exporter.export(root_node),
        }
        save_path = self.config.cache_path / write_folder / (to_name + '.json')
        UtilData.write_json_file(save_path, res, verbose=False)

    def get_fn_of_get_data(self,
                           read_data,
                           ):
        to_name_list = []
        for report in self.summary_examples:
            to_name_list.append(report.doc_id)
        read_data = set(read_data)

        def get_data(to_name, verbose=False):
            res = {}
            if 'text' in read_data or 'toc' in read_data:
                text_path = self.config.cache_path / self.processed_text_folder / (to_name + '.jsonl')
                text_jsonl = UtilData.read_raw_jsonl_file(text_path, verbose=verbose)
                toc_path = self.config.cache_path / self.processed_toc_folder / (to_name + '.jsonl')
                toc_jsonl = UtilData.read_raw_jsonl_file(toc_path, verbose=verbose)
                text_example, toc_example = self.report_parser.get_report_toc_as_example(report=text_jsonl, toc=toc_jsonl)
                res['text'] = text_example
                res['toc'] = toc_example
            if 'tree_answer' in read_data:
                tree_answer_path = self.config.cache_path / self.processed_tree_answer_folder / (to_name + '.json')
                tree_answer = UtilData.read_raw_json_file(tree_answer_path, verbose=verbose)
                res['tree_answer'] = tree_answer
            return res
        return to_name_list, get_data
