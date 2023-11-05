from cmm.trainer.basic_trainer import BasicTrainer
from cmm.conf.ESG_report_learning_conf import ESGReportLearningConfig
from cmm.optimizer.basic_optimizer import BasicOptimizer
from cmm.data_preparer.ESG_report_preparer import ESGReportPreparer
from cmm.metric.ESG_learning_metric import ESGLearningMetric
from torch.utils.data import DataLoader
from typing import List, Callable
from tqdm import tqdm
import time
import logging
import torch
from cmm.utils.tree_edit_distance import TreeEditDistanceManager


class ESGLearningTreeBasicTrainer(BasicTrainer):
    def __init__(self,
                 config: ESGReportLearningConfig,
                 model,
                 optimizer: BasicOptimizer,
                 preparer: ESGReportPreparer,
                 ):
        self.get_loader = preparer.get_fn_of_get_loader()
        loader_data = self.get_loader()
        train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader = loader_data
        super().__init__(config, model, optimizer, train_loader, dev_loader, test_loader)
        self.result_folder_path = self.result_folder_init(config.model_save_name)
        self.config = config
        self.get_loader = preparer.get_fn_of_get_loader()

    def train_batch_template(self,
                             model_run_fns: List[Callable],
                             score_fn: Callable,
                             one_split_fns: List[Callable],
                             one_merge_fns: List[Callable],
                             dataloader: DataLoader,
                             epoch=-1,
                             ):
        self.model.train()
        assert len(model_run_fns) == len(one_split_fns) == len(one_merge_fns)
        run_process_time = len(model_run_fns)
        start_time = time.time()
        raw_results: List[dict] = []
        out_of_cuda_memory_batch_count = 0
        out_of_cuda_memory_sub_batch_count = [0] * run_process_time
        sub_run_count = [0] * run_process_time
        for batch in tqdm(dataloader, unit="b", position=0, leave=True):
            out_of_memory_this_batch = False
            last_res = {}
            for run_step in range(run_process_time):
                sub_batches = one_split_fns[run_step](batch, last_res)
                res_batch = []
                for sub_batch in sub_batches:
                    sub_run_count[run_step] += 1
                    sub_res = None
                    try:
                        loss, sub_res = model_run_fns[run_step](self.model, sub_batch, run_eval=False)
                        if loss is not None:
                            loss = loss / len(sub_batches)
                            self.optimizer.gradient_update(loss)
                    except RuntimeError as exception:
                        if 'out of memory' in str(exception):
                            out_of_memory_this_batch = True
                            out_of_cuda_memory_sub_batch_count[run_step] += 1
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise exception
                    res_batch.append(sub_res)
                res = one_merge_fns[run_step](batch, last_res, sub_batches, res_batch)
                last_res = res
            if out_of_memory_this_batch:
                out_of_cuda_memory_batch_count += 1
            raw_results.append(res)
        score_to_print, score_result = score_fn(raw_results)
        used_time = (time.time() - start_time) / 60
        logging.info('Train Epoch = {}, Time = {:.2f} min, Out Of CUDA Memory = {}, Sub CUDA Memory = {}, \nScore = {}'.format(
            epoch, used_time,
            out_of_cuda_memory_batch_count / len(dataloader),
            [x / y if y != 0 else -1 for x, y in zip(out_of_cuda_memory_sub_batch_count, sub_run_count)],
            score_to_print))
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        return score_result

    def eval_batch_template(self,
                            model_run_fns: List[Callable],
                            score_fn: Callable,
                            one_split_fns: List[Callable],
                            one_merge_fns: List[Callable],
                            dataloader: DataLoader,
                            epoch=-1,
                            ):
        self.model.eval()
        assert len(model_run_fns) == len(one_split_fns) == len(one_merge_fns)
        run_process_time = len(model_run_fns)
        start_time = time.time()
        raw_results: List[dict] = []
        out_of_cuda_memory_batch_count = 0
        out_of_cuda_memory_sub_batch_count = [0] * run_process_time
        sub_run_count = [0] * run_process_time
        with torch.no_grad():
            for batch in tqdm(dataloader, unit="b", position=0, leave=True):
                out_of_memory_this_batch = False
                last_res = {}
                for run_step in range(run_process_time):
                    sub_batches = one_split_fns[run_step](batch, last_res)
                    res_batch = []
                    for sub_batch in sub_batches:
                        sub_run_count[run_step] += 1
                        try:
                            _, sub_res = model_run_fns[run_step](self.model, sub_batch, run_eval=True)
                        except RuntimeError as exception:
                            if 'out of memory' in str(exception):
                                out_of_memory_this_batch = True
                                out_of_cuda_memory_sub_batch_count[run_step] += 1
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                                continue
                            else:
                                raise exception
                        res_batch.append(sub_res)
                    res = one_merge_fns[run_step](batch, last_res, sub_batches, res_batch)
                    last_res = res
                if out_of_memory_this_batch:
                    out_of_cuda_memory_batch_count += 1
                raw_results.append(res)
        score_to_print, score_result = score_fn(raw_results)
        used_time = (time.time() - start_time) / 60
        logging.info('Eval Epoch = {}, Time = {:.2f} min, Out Of CUDA Memory = {}, Sub CUDA Memory = {}, \nScore = {}'.format(
            epoch, used_time,
            out_of_cuda_memory_batch_count / len(dataloader),
            [x / y if y != 0 else -1 for x, y in zip(out_of_cuda_memory_sub_batch_count, sub_run_count)],
            score_to_print))
        return score_result


class ESGLearningTreeTrainer(ESGLearningTreeBasicTrainer):
    def __init__(self,
                 config: ESGReportLearningConfig,
                 model,
                 optimizer: BasicOptimizer,
                 preparer: ESGReportPreparer,
                 metric: ESGLearningMetric,
                 ):
        super().__init__(config, model, optimizer, preparer)
        self.score_fn = metric.the_score_fn
        self.skip_dev = True
        self.re_set_loader = True
        self.skip_save_train_score = True

    def one_split_fn(self, batch, last_res):
        record_res, tensor_res = batch
        text_example = tensor_res['text_example']
        all_blocks = []
        for page in text_example:
            for block in page:
                block.model_size = block.size
                all_blocks.append(block)
        tensor_res['blocks'] = all_blocks

        common_size = self.model.get_common_size_of_report(tensor_res['blocks'])
        ids_to_count, ids_to_block = self.model.get_ids_from_blocks(tensor_res['blocks'])
        if 'next_node_list' in last_res:
            node_list = last_res['next_node_list']
        else:
            node_list = self.model.get_tree_nodes_with_size_from_blocks(ids_to_block, tensor_res['blocks'])
        sub_node_lists = []
        ids = []
        start_index = 0
        while start_index < len(node_list):
            sub_node_list, end_index, all_ids_to_count, valid_ids_to_count, valid_count_to_ids, valid_counts = self.model.get_sub_node_list_by_max_toward(ids_to_block, node_list, 2, self.config.max_node_per_batch, common_size, start_index)
            start_index = end_index
            if sub_node_list is None or len(sub_node_list) == 0:
                continue
            sub_node_lists.append(sub_node_list)
            ids.append({
                'all_ids_to_block': ids_to_block,
                'all_ids_to_count': all_ids_to_count,
                'valid_ids_to_count': valid_ids_to_count,
                'valid_count_to_ids': valid_count_to_ids,
                'valid_counts': valid_counts,
                'tree_answer': record_res['tree_answer']['tree_id_str'],
            })
        batch_batch = []
        for sub_node_list, id_one in zip(sub_node_lists, ids):
            new_sub_batch = (record_res, {k: v for k, v in tensor_res.items()})
            new_sub_batch[1].update({
                'node_list': sub_node_list,
                'ids': id_one,
            })
            batch_batch.append(new_sub_batch)
        return batch_batch

    def model_fn(self, model, batch: list, run_eval: bool):
        model.train()
        record_res, tensor_res = (b for b in batch)
        model_res = model.toc_rel_forward(**tensor_res)
        total_loss, res = model_res
        record = {'to_name':  record_res['to_name'],
                  'report_id': record_res['report_id'],
                  'tree_answer': record_res['tree_answer'],
                  'toc_rel_total_loss': total_loss.detach().cpu().item(),
                  }
        record.update(res)
        return total_loss, record

    def one_merge_fn(self, original_batch, last_res, sub_batch, res_batch):
        record_res, tensor_res = original_batch
        toc_rel_ans = []
        toc_rel_pred = []
        node_change_ids = {}
        for result in res_batch:
            if result is None:
                continue
            toc_rel_ans += result['toc_rel_ans']
            toc_rel_pred += result['toc_rel_pred']
            node_change_ids.update(result['node_change_ids'])
        ids_to_count, ids_to_block = self.model.get_ids_from_blocks(tensor_res['blocks'])
        common_size = self.model.get_common_size_of_report(tensor_res['blocks'])
        if 'next_node_list' in last_res:
            node_list = last_res['next_node_list']
        else:
            node_list = self.model.get_tree_nodes_with_size_from_blocks(ids_to_block, tensor_res['blocks'], common_size)
        node_list_tree_str = TreeEditDistanceManager.convert_tree_to_string(node_list[0], value_fn=lambda x: x.name, children_fn=lambda x: x.children)
        toc_rel_pred_node_list = self.model.get_tree_nodes_with_size_from_node_list_with_node_change(ids_to_block, node_list, node_change_ids, common_size=common_size)
        toc_tel_pred_tree_str = TreeEditDistanceManager.convert_tree_to_string(toc_rel_pred_node_list[0], value_fn=lambda x: x.name, children_fn=lambda x: x.children)
        ave_loss = [x['toc_rel_total_loss'] for x in res_batch if x is not None]
        ave_loss = sum(ave_loss) if len(ave_loss) > 0 else 0
        ave_loss = ave_loss / len(res_batch) if len(res_batch) > 0 else 0
        last_res.update({
            'to_name': record_res['to_name'],
            'report_id': record_res['report_id'],
            'tree_answer': record_res['tree_answer'],
            'total_loss': ave_loss,
            'toc_rel_ans': toc_rel_ans,
            'toc_rel_pred': toc_rel_pred,
            'toc_rel_pred_node_list': toc_rel_pred_node_list,
            'rel_total_loss': [ave_loss],
            'toc_rel_pred_tree_str': toc_tel_pred_tree_str,
            'next_node_list': toc_rel_pred_node_list,
            'node_list_tree_str': node_list_tree_str,
        })
        return last_res

    def train(self):
        one_split_fn = self.one_split_fn
        train_eval_args = {
            'model_run_fns': [self.model_fn],
            'score_fn': self.score_fn,
            'one_split_fns': [one_split_fn],
            'one_merge_fns': [self.one_merge_fn],
        }
        self.basic_train_template(train_batch_fn=self.train_batch_template,
                                  train_args=train_eval_args,
                                  eval_batch_fn=self.eval_batch_template,
                                  eval_args=train_eval_args,
                                  )
