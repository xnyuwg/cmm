from typing import List
from cmm.data_preparer.ESG_report_preparer import ESGReportPreparer
import time
from cmm.utils.tree_edit_distance import TreeEditDistanceManager
from cmm.conf.ESG_report_learning_conf import ESGReportLearningConfig
from cmm.utils.util_structure import UtilStructure


class ESGLearningMetric:
    def __init__(self,
                 config: ESGReportLearningConfig,
                 preparer: ESGReportPreparer):
        self.config = config
        self.to_name_list = preparer.to_name_list
        self.get_data_fn = preparer.get_data_fn
        self.report_count2id = self.to_name_list
        self.report_id2count = {t: i for i, t in enumerate(self.to_name_list)}

    def the_score_fn(self, results: List[dict]):
        start_time = time.time()
        # loss
        if 'total_loss' in results[0]:
            total_num = len(results)
            mean_loss = sum([x['total_loss'] for x in results]) / total_num if total_num != 0 else -1
            loss_to_print = "Loss = {:.4f}, ".format(mean_loss)
        else:
            loss_to_print, mean_loss = 'None', None

        # label score
        if UtilStructure.if_any_element_of_alist_in_blist(['toc_rel_ans', 'toc_rel_pred'], results[0]):
            toc_rel_ans = [x['toc_rel_ans'] for x in results]
            toc_rel_pred = [x['toc_rel_pred'] for x in results]
            toc_rel_label_to_print, toc_rel_label_score_results = self.label_score_for_one(toc_rel_ans, toc_rel_pred)
        else:
            toc_rel_label_to_print, toc_rel_label_score_results = 'None', None

        # teds score
        if 'toc_rel_pred_tree_str' in results[0]:
            tree_answer = [x['tree_answer'] for x in results]
            toc_rel_tree_pred = [x['toc_rel_pred_tree_str'] for x in results]
            teds_from_tree_to_print, teds_from_tree_score_results = self.teds_tree_score_fn(tree_answer, toc_rel_tree_pred)
        else:
            teds_from_tree_to_print, teds_from_tree_score_results = 'None', None

        used_time = (time.time() - start_time) / 60
        final_to_print = loss_to_print + '\n' \
                         + toc_rel_label_to_print \
                         + '\n' + teds_from_tree_to_print
        final_score_results = {"time": used_time,
                               'loss': mean_loss,
                               'HD': toc_rel_label_score_results,
                               'TEDS': teds_from_tree_score_results,
                               }
        return final_to_print, final_score_results

    def label_score_for_one(self, ans, pred):
        assert len(ans) == len(pred)
        total_ans = []
        total_pred = []
        for an, pr in zip(ans, pred):
            total_ans += an
            total_pred += pr
        total_ans = [x if x != 2 else 0 for x in total_ans]
        total_pred = [x if x != 2 else 0 for x in total_pred]
        assert len(total_ans) == len(total_pred)
        p = len([i for i in range(len(total_ans)) if total_pred[i] == 0 and total_ans[i] == 0]) / len([i for i in range(len(total_ans)) if total_pred[i] == 0]) if len([i for i in range(len(total_ans)) if total_pred[i] == 0]) > 0 else 0
        r = len([i for i in range(len(total_ans)) if total_pred[i] == 0 and total_ans[i] == 0]) / len([i for i in range(len(total_ans)) if total_ans[i] == 0]) if len([i for i in range(len(total_ans)) if total_ans[i] == 0]) > 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        score_results = {'p': p,
                         'r': r,
                         'f1': f1,
                         }
        to_print = "HD: p={:.3f} r={:.3f} f1={:.3f}".format(p, r, f1)
        return to_print, score_results

    def teds_tree_score_fn(self, ans_tree, pred_tree):
        start_time = time.time()
        assert len(ans_tree) == len(pred_tree)
        teds = []
        for ans, pred in zip(ans_tree, pred_tree):
            ans = ans['tree_id_str']
            sim = TreeEditDistanceManager.tree_edit_distance_similarity_with_string(tree1=ans, tree2=pred)
            teds.append(sim)
        final_sim = sum(teds) / len(teds)
        used_time = (time.time() - start_time) / 60
        score_results = {'time': used_time,
                         'teds': final_sim,
                         }
        to_print = "TEDS: Sim = {:.3f}".format(final_sim)
        return to_print, score_results
