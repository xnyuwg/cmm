import logging
import torch
from torch import nn
from cmm.model.basic_model import BasicModel
from cmm.data_processor.ESG_report_processor import ESGReportProcessor
from cmm.conf.ESG_report_learning_conf import ESGReportLearningConfig
import torch_geometric.nn
from anytree import Node, LevelOrderIter
from cmm.utils.util_structure import UtilStructure
from cmm.utils.util_data import UtilData
from cmm.utils.tree_edit_distance import TreeEditDistanceManager
from anytree.importer import JsonImporter


class ESGGRU(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.dropout_ratio = dropout_ratio
        lstms = [nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim // 2, num_layers=1, dropout=dropout_ratio, bidirectional=True) for _ in range(layer_num)]
        self.lstms = nn.ModuleList(lstms)
        linears = [nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
        ) for _ in range(layer_num)]
        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        for i in range(self.layer_num):
            x_o, _ = self.lstms[i](x)
            x_o = self.linears[i](x_o)
            x = x + x_o
        return x


class ESGGAT2Conv(nn.Module):
    def __init__(self, node_dim, edge_attr_dim, layer_num, dropout_ratio, head=None):
        super().__init__()
        self.layer_num = layer_num
        self.dropout_ratio = dropout_ratio
        gcns = [torch_geometric.nn.GATv2Conv(in_channels=node_dim, out_channels=node_dim // head, dropout=dropout_ratio, edge_dim=edge_attr_dim, heads=head) for _ in range(layer_num)]
        self.gcns = nn.ModuleList(gcns)
        linears = [nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(node_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
        ) for _ in range(layer_num)]
        self.linears = nn.ModuleList(linears)

    def forward(self, x, edge_index, edge_type=None, edge_attr=None, node_type=None, edge_weight=None):
        for i in range(self.layer_num):
            x_o = self.gcns[i](x=x, edge_index=edge_index, edge_attr=edge_attr)
            x_o = self.linears[i](x_o)
            x = x + x_o
        return x


class ESGTocRelWeight(nn.Module):
    def __init__(self, config, hidden_sen_emb_dim, dropout_ratio, input_sen_emb_dim, b_fea_dim, edge1_type_num, edge1_type_dim, edge1_attr_dim, edge1_attr_hidden_dim, graph1_layer_num, font_num, font_dim):
        super().__init__()
        self.label_toc_logit_linear = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_sen_emb_dim, hidden_sen_emb_dim // 4),
            nn.LayerNorm(hidden_sen_emb_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_sen_emb_dim // 4, 2),
        )

        self.label_rel_combine_logit_linear = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_sen_emb_dim * 2, hidden_sen_emb_dim // 4),
            nn.LayerNorm(hidden_sen_emb_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_sen_emb_dim // 4, 1),
        )

        self.language_model = BasicModel.new_auto_model(config.lm_model_name)

        self.edge1_type_embedding = nn.Embedding(num_embeddings=edge1_type_num, embedding_dim=edge1_type_dim)

        self.sen_emb_fea_linear = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(input_sen_emb_dim + b_fea_dim + font_dim, hidden_sen_emb_dim),
            nn.LayerNorm(hidden_sen_emb_dim),
        )

        self.b_fea_linear = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(b_fea_dim, b_fea_dim),
            nn.LayerNorm(b_fea_dim),
        )

        self.edge1_linear = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(edge1_attr_dim, edge1_attr_hidden_dim),
            nn.LayerNorm(edge1_attr_hidden_dim),
        )

        self.lstm1 = ESGGRU(hidden_dim=hidden_sen_emb_dim, dropout_ratio=dropout_ratio, layer_num=1)

        self.gcn1 = ESGGAT2Conv(node_dim=hidden_sen_emb_dim, edge_attr_dim=edge1_attr_hidden_dim, dropout_ratio=dropout_ratio, layer_num=graph1_layer_num, head=4)


class ESGTreeModel(BasicModel):
    def __init__(self,
                 config: ESGReportLearningConfig,
                 processor: ESGReportProcessor,
                 ):
        super().__init__(config=config)
        self.config = config
        logging.info('in model common_size_chose={}'.format(config.common_size_chose))
        self.slow_para = ['language_model']
        self.input_sen_emb_dim = config.sentence_lm_embedding_dim
        self.hidden_sen_emb_dim = config.hidden_sen_emb_dim
        self.dropout_ratio = 0.15
        self.b_fea_dim = 14
        self.edge1_type_dim = 3
        self.edge1_type_num = 3
        self.edge1_attr_dim = 10 + self.edge1_type_dim
        self.edge1_attr_hidden_dim = 4
        self.graph1_layer_num = 2
        self.font_list = processor.font_list
        self.font_to_id = {x: i for i, x in enumerate(self.font_list)}
        self.font_dim = 0
        self.label_map = {0: 'keep', 1: 'delete', 2: 'move'}

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        self.ce_none_reduction_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=self.config.label_smoothing)

        self.model_para = ESGTocRelWeight(config=config,
                                          hidden_sen_emb_dim=self.hidden_sen_emb_dim,
                                          dropout_ratio=self.dropout_ratio,
                                          input_sen_emb_dim=self.input_sen_emb_dim,
                                          b_fea_dim=self.b_fea_dim,
                                          edge1_type_num=self.edge1_type_num,
                                          edge1_type_dim=self.edge1_type_dim,
                                          edge1_attr_dim=self.edge1_attr_dim,
                                          edge1_attr_hidden_dim=self.edge1_attr_hidden_dim,
                                          graph1_layer_num=self.graph1_layer_num,
                                          font_num=len(self.font_list),
                                          font_dim=self.font_dim)

    def get_common_size_of_report(self, blocks, common_size_chose=None):
        if common_size_chose is None:
            common_size_chose = self.config.common_size_chose
        if common_size_chose is None:
            raise Exception('common_size_chose is None!!!')
        size_count = {}
        for block in blocks:
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
        if step < common_size_chose:
            the_size = the_size - 1
        return the_size

    def get_ids_from_blocks(self, blocks):
        ids_to_block = {}
        ids_to_count = {}
        count = -1
        for block in blocks:
            count += 1
            ids_to_count[block.id] = count
            ids_to_block[block.id] = block
        return ids_to_count, ids_to_block

    def get_sub_node_list_by_max_toward(self, use_ids_to_block, node_list, toward_num, max_node_num, common_size, start_index, use_ids=None):
        valid_ids = set()
        all_ids = set()
        end_index = start_index
        for i in range(start_index, len(node_list)):
            node = node_list[i]
            block = use_ids_to_block[node.name]
            if common_size is not None and block.size <= common_size:
                continue
            if use_ids is not None and node.name not in use_ids:
                continue
            current_toward_num = toward_num
            end_node_add = False
            try_next_toward_num = True
            while current_toward_num >= 1 and try_next_toward_num:
                total_candidate_id = set([node.name])
                prev_candidate = [node]
                for _ in range(current_toward_num):
                    next_candidate = []
                    for pc in prev_candidate:
                        parent_node = pc.parent
                        children_nodes = pc.children
                        pc_as_child_index = None
                        sibling_nodes = []
                        if pc.parent is not None:
                            for j, child in enumerate(pc.parent.children):
                                if child.name == pc.name:
                                    pc_as_child_index = j
                                    break
                            if pc_as_child_index is not None and pc_as_child_index > 0:
                                sibling_nodes.append(pc.parent.children[pc_as_child_index - 1])
                            if pc_as_child_index is not None and pc_as_child_index < len(pc.parent.children) - 1:
                                sibling_nodes.append(pc.parent.children[pc_as_child_index + 1])
                            next_candidate += [parent_node]
                        next_candidate += list(children_nodes) + sibling_nodes
                    total_candidate_id.update([x.name for x in next_candidate])
                    prev_candidate = next_candidate

                if len(all_ids) + len(total_candidate_id) <= max_node_num:
                    all_ids.update(total_candidate_id)
                    valid_ids.add(node.name)
                    end_index = i
                    break
                else:
                    if len(valid_ids) == 0:
                        current_toward_num -= 1
                        try_next_toward_num = True
                        if current_toward_num <= 0:
                            # logging.warning('node {} too long {}, skip'.format(node.name, len(total_candidate_id)))
                            all_ids.update(node.name)
                            valid_ids.add(node.name)
                            end_index = i
                    else:
                        try_next_toward_num = False
                        end_node_add = True
                        break
            if end_node_add:
                break
        end_index += 1
        all_ids.update(valid_ids)
        sub_node_list = []
        all_ids_to_count = {}
        valid_ids_to_count = {}
        count = -1
        for node in node_list:
            if node.name in all_ids:
                count += 1
                sub_node_list.append(node)
                all_ids_to_count[node.name] = count
                if node.name in valid_ids:
                    valid_ids_to_count[node.name] = count
        valid_counts = sorted([v for k, v in valid_ids_to_count.items()])
        valid_count_to_ids = {v: k for k, v in valid_ids_to_count.items()}
        return sub_node_list, end_index, all_ids_to_count, valid_ids_to_count, valid_count_to_ids, valid_counts

    def get_tree_nodes_with_size_from_blocks(self, ids_to_block, blocks, common_size=None):
        node_list = []
        for block in blocks:
            if common_size is None or (common_size is not None and block.size > common_size):
                node = Node(block.id)
                node_list.append(node)

        for t_count, t_node in enumerate(node_list):
            t_block = ids_to_block[t_node.name]
            for h_count in reversed(range(t_count)):
                h_node = node_list[h_count]
                h_block = ids_to_block[h_node.name]
                if h_block.size > t_block.size or h_block.id == 'r0':
                    t_node.parent = h_node
                    break
        return node_list

    def get_tree_nodes_with_size_from_node_list_with_node_change(self, ids_to_block, old_node_list, node_change_ids, common_size=None):
        self_id_not_in_printed = False
        new_node_list = []
        node_change_ids['r0'] = 'keep'
        for old_node in old_node_list:
            block = ids_to_block[old_node.name]
            if common_size is None or (common_size is not None and block.size > common_size):
                if block.id not in node_change_ids:
                    if not self_id_not_in_printed:
                        logging.error('self_id not in node_change_ids: id={}, len(node_change_ids)={}, len(old_node_list)={}'.format(block.id, len(node_change_ids), len(old_node_list)))
                        self_id_not_in_printed = True
                    continue
                if node_change_ids[block.id] == 'delete':
                    continue
                new_node = Node(block.id)
                new_node_list.append(new_node)

        for t_count, t_node in enumerate(new_node_list):
            t_block = ids_to_block[t_node.name]
            for h_count in reversed(range(t_count)):
                h_node = new_node_list[h_count]
                h_block = ids_to_block[h_node.name]
                if h_block.size > t_block.size or h_block.id == 'r0':
                    t_node.parent = h_node
                    break

        for node in new_node_list:
            if node_change_ids[node.name] == 'move':
                prev_node, next_node = UtilStructure.get_a_node_prev_next_siblings_in_tree(node)
                if prev_node is None:
                    continue
                node.parent = prev_node
                for child in node.children:
                    child.parent = prev_node
        return new_node_list

    def get_tree_nodes_with_size_from_node_list_when_correct(self, ids_to_block, old_node_list, tree_answer):
        importer = JsonImporter()
        ans_root = importer.import_(tree_answer['tree_node_list'])
        ans_node_list = list(LevelOrderIter(ans_root))
        ans_ids = set([node.name for node in ans_node_list])
        ans_ids.add('r0')

        first_node_list = []
        for old_node in old_node_list:
            if old_node.name in ans_ids:
                new_node = Node(old_node.name)
                first_node_list.append(new_node)

        for t_count, t_node in enumerate(first_node_list):
            t_block = ids_to_block[t_node.name]
            for h_count in reversed(range(t_count)):
                h_node = first_node_list[h_count]
                h_block = ids_to_block[h_node.name]
                if h_block.size > t_block.size or h_block.id == 'r0':
                    t_node.parent = h_node
                    break

        current_node_list = first_node_list
        current_score = TreeEditDistanceManager.tree_edit_distance_similarity_with_string(tree1=tree_answer['tree_id_str'], tree2=TreeEditDistanceManager.convert_tree_to_string(current_node_list[0], value_fn=lambda x: x.name, children_fn=lambda x: x.children))
        for node in reversed(first_node_list):
            if node.name == 'r0':
                continue
            last_score = current_score
            last_node_list = current_node_list

            delete_node_list = []
            for last_node in last_node_list:
                if last_node.name != node.name:
                    new_node = Node(last_node.name)
                    delete_node_list.append(new_node)
            for t_count, t_node in enumerate(delete_node_list):
                t_block = ids_to_block[t_node.name]
                for h_count in reversed(range(t_count)):
                    h_node = delete_node_list[h_count]
                    h_block = ids_to_block[h_node.name]
                    if h_block.size > t_block.size or h_block.id == 'r0':
                        t_node.parent = h_node
                        break
            delete_score = TreeEditDistanceManager.tree_edit_distance_similarity_with_string(tree1=tree_answer['tree_id_str'], tree2=TreeEditDistanceManager.convert_tree_to_string(delete_node_list[0], value_fn=lambda x: x.name, children_fn=lambda x: x.children))

            move_score = 0
            original_size = ids_to_block[node.name].size
            if node.parent is None:
                move_score = -1
            prev_node, next_node = UtilStructure.get_a_node_prev_next_siblings_in_tree(node)
            if prev_node is None:
                move_score = -1
            if move_score != -1:
                prev_children = prev_node.children
                node_children = node.children
                if len(prev_children) > 0:
                    prev_child = prev_children[-1]
                    should_size = ids_to_block[prev_child.name].size
                elif len(node_children) > 0 and ids_to_block[node_children[0].name].size < ids_to_block[prev_node.name].size:
                    should_size = ids_to_block[node_children[0].name].size
                else:
                    should_size = ids_to_block[prev_node.name].size - 1
                ids_to_block[node.name].size = should_size
                move_node_list = []
                for last_node in last_node_list:
                    new_node = Node(last_node.name)
                    move_node_list.append(new_node)
                for t_count, t_node in enumerate(move_node_list):
                    t_block = ids_to_block[t_node.name]
                    for h_count in reversed(range(t_count)):
                        h_node = move_node_list[h_count]
                        h_block = ids_to_block[h_node.name]
                        if h_block.size > t_block.size or h_block.id == 'r0':
                            t_node.parent = h_node
                            break
                move_score = TreeEditDistanceManager.tree_edit_distance_similarity_with_string(tree1=tree_answer['tree_id_str'], tree2=TreeEditDistanceManager.convert_tree_to_string(move_node_list[0], value_fn=lambda x: x.name, children_fn=lambda x: x.children))

            max_score, max_choice = UtilStructure.find_max_and_number_index([last_score, delete_score, move_score])
            if max_choice == 0:
                current_score = last_score
                current_node_list = last_node_list
                ids_to_block[node.name].size = original_size
            elif max_choice == 1:
                current_score = delete_score
                current_node_list = delete_node_list
                ids_to_block[node.name].size = original_size
            elif max_choice == 2:
                current_score = move_score
                current_node_list = move_node_list
                ids_to_block[node.name].size = should_size
        return current_node_list

    def get_valid_list_ids(self, ids):
        valid_list_count_to_valid_count = {}
        valid_count_to_valid_list_count = {}
        for i, count in enumerate(ids['valid_counts']):
            valid_list_count_to_valid_count[i] = count
            valid_count_to_valid_list_count[count] = i
        valid_ids_to_valid_list_count = {k: valid_count_to_valid_list_count[v] for k, v in ids['valid_ids_to_count'].items()}
        valid_valid_list_count_to_ids = {valid_count_to_valid_list_count[k]: v for k, v in ids['valid_count_to_ids'].items()}
        valid_list_ids = {
            'valid_list_count_to_valid_count': valid_list_count_to_valid_count,
            'valid_count_to_valid_list_count': valid_count_to_valid_list_count,
            'valid_ids_to_valid_list_count': valid_ids_to_valid_list_count,
            'valid_list_count_to_valid_ids': valid_valid_list_count_to_ids,
        }
        return valid_list_ids

    def get_node_tree_level(self, ids_to_level, node, upper_level):
        if node is None:
            return
        current_level = upper_level + 1
        ids_to_level[node.name] = current_level
        children = node.children
        for child in children:
            self.get_node_tree_level(ids_to_level, child, current_level)

    def get_move_same_level_ids(self, node_list, ids):
        move_ids = []
        for node in node_list:
            move_id = []
            if node.name not in ids['valid_ids_to_count']:
                continue
            if node.parent is not None:
                for child in node.parent.children:
                    if child.name == node.name:
                        break
                    if child.name in ids['all_ids_to_count']:
                        move_id.append(child.name)
            move_ids.append(move_id)
        return move_ids

    def get_node_up_down_label_with_toc_example(self, valid_ids_to_count, node_list, toc_example):
        toc_ids_to_level = {x[3]: x[0] for x in toc_example}
        rel_up_down_label_cls = [0] * len(valid_ids_to_count)
        for node in node_list:
            if node.name in valid_ids_to_count and node.name in toc_ids_to_level:
                before_nodes = []
                if node.parent is not None:
                    for child in node.parent.children:
                        if child.name == node.name:
                            break
                        before_nodes.append(child)
                if len(before_nodes) == 0:
                    continue
                for be_node in before_nodes:
                    if be_node.name in toc_ids_to_level and toc_ids_to_level[be_node.name] < toc_ids_to_level[node.name]:
                        rel_up_down_label_cls[valid_ids_to_count[node.name]] = 2
                        break
        return rel_up_down_label_cls

    def remove_nodes_in_a_tree(self, ids_to_count, node_list, nodes_keep_id):
        nodes_keep_id = set(nodes_keep_id + ['r0'])
        new_node_list = []
        for node in node_list:
            if node.name in nodes_keep_id:
                new_node_list.append(node)
            else:
                node_parent = node.parent
                # if node_parent is not None:
                new_node_children = [child for child in node_parent.children if child.name != node.name]
                node_parent.children = new_node_children
                for child in node.children:
                    child.parent = node_parent

        for node in node_list:
            children = node.children
            children = sorted(children, key=lambda x: int(ids_to_count[x.name]))
            node.children = children
        return new_node_list

    def change_nodes_up_down_in_a_tree(self, ids_to_count, node_list, rel_up_down_ids):
        for node in node_list:
            if node.name in rel_up_down_ids and node.name != 'r0':
                if rel_up_down_ids[node.name] == 1:
                    if node.parent is not None and node.parent.parent is not None and node.parent.parent.children is not None:
                        node_parent_parent = node.parent.parent
                        node.parent = node_parent_parent
                        node_parent_parent.children = sorted(node_parent_parent.children, key=lambda x: ids_to_count[x.name])
                if rel_up_down_ids[node.name] == 2:
                    prev_node, next_node = UtilStructure.get_a_node_prev_next_siblings_in_tree(node)
                    if prev_node is None:
                        to_parent_node = node.parent
                    else:
                        to_parent_node = prev_node
                    if to_parent_node is not None:
                        node.parent = to_parent_node
                        if node.children is not None:
                            for child in node.children:
                                child.parent = to_parent_node
                        to_parent_node.children = sorted(to_parent_node.children, key=lambda x: ids_to_count[x.name])
        return node_list

    def get_b_fea_from_text(self, blocks, common_size):
        max_size = max([x.size for x in blocks])
        features = []
        for block in blocks:
            rel_size = block.size / max_size
            abs_size = block.size / common_size / 10 if common_size is not None and common_size > 0 else block.size / 10
            abs_page = block.page / 100
            rel_xy_cut_read_order_page = block.xy_cut_sequence_number / 100
            rel_x1 = block.position[0] / block.page_width
            rel_y1 = block.position[1] / block.page_height
            rel_x2 = block.position[2] / block.page_width
            rel_y2 = block.position[3] / block.page_height
            r, g, b = UtilData.decimal_to_rgb(int(block.color))
            r = r / 255
            g = g / 255
            b = b / 255
            text_char_len = len(block.text) / 10000
            text_word_len = len(block.text.split(' ')) / 1000
            text_lines = len(block.original_text) / 10
            fea = [rel_size, abs_size, abs_page, rel_xy_cut_read_order_page, rel_x1, rel_y1, rel_x2, rel_y2, r, g, b, text_char_len, text_word_len, text_lines]
            features.append(fea)
        return features

    def get_b_fea_from_node_list(self, node_list, ids_to_block, common_size):
        blocks = [ids_to_block[x.name] for x in node_list]
        return self.get_b_fea_from_text(blocks, common_size)

    def get_token_from_node_list(self, node_list, ids_to_block):
        tokenizer = self.get_tokenizer()
        text_strs = []
        for node in node_list:
            block = ids_to_block[node.name]
            text_strs.append(block.text)
        token_res = tokenizer(text_strs, add_special_tokens=True, max_length=self.config.sentence_token_max_length, padding='longest', truncation=True, return_attention_mask=True)
        token_ids, token_mask = token_res.input_ids, token_res.attention_mask
        return token_ids, token_mask

    def set_edge_count(self, edge_index_set, edge_index, edge_type, edge_attr, h_block, t_block, h_count, t_count, edge_kind, common_size):
        if (h_count, t_count) not in edge_index_set:
            abs_page_dis = (h_block.page - t_block.page) / 10
            x1_dis = (h_block.position[0] - t_block.position[0]) / max(h_block.page_width, t_block.page_width)
            y1_dis = (h_block.position[1] - t_block.position[1]) / max(h_block.page_height, t_block.page_height)
            x2_dis = (h_block.position[2] - t_block.position[2]) / max(h_block.page_width, t_block.page_width)
            y2_dis = (h_block.position[3] - t_block.position[3]) / max(h_block.page_height, t_block.page_height)
            rel_size_dis = (h_block.size - t_block.size) / common_size / 10 if common_size is not None and common_size > 0 else (t_block.size - h_block.size) / 100
            abs_size_dis = (h_block.size - t_block.size) / 100
            size_compare = 0
            if h_block.size > t_block.size:
                size_compare = 1
            if h_block.size < t_block.size:
                size_compare = -1
            font_same = 1 if h_block.font == t_block.font else 0
            color_same = 1 if h_block.color == t_block.color else 0
            edge_index.append([h_count, t_count])
            edge_type.append(edge_kind)
            attr = [abs_page_dis, rel_size_dis, abs_size_dis, size_compare, x1_dis, y1_dis, x2_dis, y2_dis, font_same, color_same]
            edge_attr.append(attr)
            edge_index_set.add((h_count, t_count))

    def get_edge_from_node_list(self, toward_num, node_list, use_ids_to_block, use_ids_to_count, valid_ids_to_count, common_size):
        edge_kind_start_offset = None
        edge_index = []
        edge_type = []
        edge_attr = []
        edge_index_set = set()
        edge_kind_start_offset = edge_kind_start_offset + 3 if edge_kind_start_offset is not None else 0

        for node in node_list:
            if node.name not in valid_ids_to_count:
                continue
            prev_candidate = [node]
            for _ in range(toward_num):
                next_candidate = []
                for pc in prev_candidate:
                    parent_nodes = [pc.parent] if pc.parent is not None else []
                    children_nodes = pc.children
                    pc_as_child_index = None
                    sibling_nodes = []
                    if pc.parent is not None:
                        for j, child in enumerate(pc.parent.children):
                            if child.name == pc.name:
                                pc_as_child_index = j
                                break
                        if pc_as_child_index is not None and pc_as_child_index > 0:
                            sibling_nodes.append(pc.parent.children[pc_as_child_index - 1])
                        if pc_as_child_index is not None and pc_as_child_index < len(pc.parent.children) - 1:
                            sibling_nodes.append(pc.parent.children[pc_as_child_index + 1])
                    parent_nodes = [x for x in parent_nodes if x.name in use_ids_to_count]
                    for pn in parent_nodes:
                        self.set_edge_count(edge_index_set=edge_index_set, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr, h_block=use_ids_to_block[pn.name], t_block=use_ids_to_block[pc.name],
                                            h_count=use_ids_to_count[pn.name], t_count=use_ids_to_count[pc.name], edge_kind=0 + edge_kind_start_offset, common_size=common_size)
                    children_nodes = [x for x in list(children_nodes) if x.name in use_ids_to_count]
                    for cn in children_nodes:
                        self.set_edge_count(edge_index_set=edge_index_set, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr, h_block=use_ids_to_block[cn.name], t_block=use_ids_to_block[pc.name],
                                            h_count=use_ids_to_count[cn.name], t_count=use_ids_to_count[pc.name], edge_kind=1 + edge_kind_start_offset, common_size=common_size)
                    sibling_nodes = [x for x in sibling_nodes if x.name in use_ids_to_count]
                    for sn in sibling_nodes:
                        self.set_edge_count(edge_index_set=edge_index_set, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr, h_block=use_ids_to_block[sn.name], t_block=use_ids_to_block[pc.name],
                                            h_count=use_ids_to_count[sn.name], t_count=use_ids_to_count[pc.name], edge_kind=2 + edge_kind_start_offset, common_size=common_size)
                    next_candidate += parent_nodes + list(children_nodes) + sibling_nodes
                prev_candidate = next_candidate
        return edge_index, edge_type, edge_attr

    def get_toc_ans_from_toc_example(self, toc_example, valid_ids_to_count):
        toc_ans = [1] * len(valid_ids_to_count)
        for toc in toc_example + [[None, None, None, 'r0']]:
            self_id = toc[3]
            if self_id in valid_ids_to_count:
                toc_ans[valid_ids_to_count[self_id]] = 0
        return toc_ans

    def get_keep_delete_up_down_node_ids(self, max_index, valid_count_to_ids):
        res = {}
        for i, x in enumerate(max_index):
            res[valid_count_to_ids[i]] = self.label_map[x]
        return res

    def feature_b_cat_forward(self,
                              use_model,
                              sen_emb: torch.Tensor = None,
                              b_fea: torch.Tensor = None,
                              font_fea_tensor: torch.Tensor = None,
                              ):
        # (block_num, b_fea)
        b_fea = use_model.b_fea_linear(b_fea)
        to_cat = [sen_emb]
        if b_fea is not None:
            to_cat.append(b_fea)
        if font_fea_tensor is not None:
            to_cat.append(font_fea_tensor)
        sen_emb = torch.cat(to_cat, dim=1)
        # (block_num, sen_emb_dim)
        sen_emb = use_model.sen_emb_fea_linear(sen_emb)
        return sen_emb

    def sentence_lm_forward(self,
                            use_model,
                            sen_token_ids: torch.Tensor = None,
                            sen_token_mask: torch.Tensor = None):
        # (block_num, sen_token_len, lm_dim)
        lm_out = use_model.language_model(input_ids=sen_token_ids, attention_mask=sen_token_mask)
        # (block_num, lm_dim)
        cls_out = lm_out.last_hidden_state[:, 0]
        return cls_out

    def toc_rel_logit_forward(self,
                              all_ids_to_count,
                              valid_block_emb_tensor,
                              block_emb_tensor,
                              move_ids):
        # (block_num, 2)
        toc_logit_tensor = self.model_para.label_toc_logit_linear(valid_block_emb_tensor)

        moves_tensor = []
        assert valid_block_emb_tensor.size(0) == len(move_ids)
        for i in range(valid_block_emb_tensor.size(0)):
            valid_tensor = valid_block_emb_tensor[i]
            move_id = move_ids[i]
            move_id = move_id[-4:]
            move_counts = [all_ids_to_count[x] for x in move_id]
            if move_counts is not None and len(move_counts) > 0:
                # (num, dim)
                move_tensor = block_emb_tensor[move_counts]
            else:
                move_tensor = torch.zeros_like(valid_tensor)
                move_tensor = move_tensor.unsqueeze(0)
            # (1, dim)
            move_att_tensor, _ = torch.max(move_tensor, dim=0)
            move_att_tensor = move_att_tensor.unsqueeze(0)
            moves_tensor.append(move_att_tensor)

        moves_tensor = [x for x in moves_tensor]
        moves_tensor = torch.cat(moves_tensor, dim=0)

        move_combine_tensor = torch.cat([valid_block_emb_tensor, moves_tensor], dim=1)

        move_logit_tensor = self.model_para.label_rel_combine_logit_linear(move_combine_tensor)

        # (block_num, 3)
        logit_tensor = torch.cat([toc_logit_tensor, move_logit_tensor], dim=1)
        return logit_tensor

    def logit_loss_forward(self,
                           logit: torch.Tensor = None,
                           ans: torch.Tensor = None):
        loss = 0
        if ans is not None:
            loss = self.ce_loss(logit, ans)
        return loss

    def before_emb_forward(self,
                           use_model=None,
                           node_list=None,
                           ids_to_block=None,
                           common_size=None,
                           ):
        # fea
        b_fea = self.get_b_fea_from_node_list(node_list, ids_to_block, common_size)

        # LM
        # (block_num, sen_len)
        block_token_ids, block_token_mask = self.get_token_from_node_list(node_list, ids_to_block)
        block_token_ids_tensor = torch.tensor(block_token_ids, dtype=torch.long, device=self.get_model_device())
        block_token_mask_tensor = torch.tensor(block_token_mask, dtype=torch.long, device=self.get_model_device())
        # (block_num, sen_dim)
        block_emb_tensor = self.sentence_lm_forward(use_model=use_model,
                                                    sen_token_ids=block_token_ids_tensor,
                                                    sen_token_mask=block_token_mask_tensor)

        b_fea_tensor = torch.tensor(b_fea, dtype=torch.float, device=self.get_model_device())
        block_emb_tensor = self.feature_b_cat_forward(use_model, block_emb_tensor, b_fea_tensor)
        return block_emb_tensor

    def graph1_forward(self,
                       node_list,
                       ids,
                       common_size):
        block_emb_tensor = self.before_emb_forward(use_model=self.model_para, node_list=node_list, ids_to_block=ids['all_ids_to_block'], common_size=common_size)

        block_emb_tensor = self.model_para.lstm1(block_emb_tensor)

        edge_index, edge_type, edge_attr = self.get_edge_from_node_list(2, node_list, ids['all_ids_to_block'], ids['all_ids_to_count'], ids['valid_ids_to_count'], common_size)
        if len(edge_index) == 0:
            # logging.warning("graph1_forward len edge_index is 0! len_node_list={}, len valid_counts={}, node.name={}".format(len(node_list), len(ids['valid_counts']), [x.name for x in node_list]))
            return block_emb_tensor
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.get_model_device())
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long, device=self.get_model_device())
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float, device=self.get_model_device())
        edge_index_tensor = edge_index_tensor.transpose(0, 1)
        edge_type_emb_tensor = self.model_para.edge1_type_embedding(edge_type_tensor)
        edge_attr_tensor = torch.cat([edge_type_emb_tensor, edge_attr_tensor], dim=1)
        edge_attr_tensor = self.model_para.edge1_linear(edge_attr_tensor)
        block_emb_tensor = self.model_para.gcn1(x=block_emb_tensor, edge_index=edge_index_tensor, edge_type=edge_type_tensor, edge_attr=edge_attr_tensor)
        return block_emb_tensor

    def toc_rel_forward(self,
                        text_example=None,
                        toc_example=None,
                        blocks=None,
                        node_list=None,
                        ids=None,
                        ):

        # init
        common_size = self.get_common_size_of_report(blocks)
        valid_list_ids = self.get_valid_list_ids(ids)
        move_ids = self.get_move_same_level_ids(node_list, ids)

        # graph1 forward
        block_emb_tensor = self.graph1_forward(node_list=node_list, ids=ids, common_size=common_size)
        valid_block_emb_tensor = block_emb_tensor[ids['valid_counts']]

        # logit
        logit_tensor = self.toc_rel_logit_forward(ids['all_ids_to_count'], valid_block_emb_tensor, block_emb_tensor, move_ids)
        prob_tensor = torch.softmax(logit_tensor, dim=1)
        max_prob_tensor, max_index_tensor = torch.max(prob_tensor, dim=1)

        # loss
        toc_ans = self.get_toc_ans_from_toc_example(toc_example, valid_list_ids['valid_ids_to_valid_list_count'])
        rel_ans = self.get_node_up_down_label_with_toc_example(valid_list_ids['valid_ids_to_valid_list_count'], node_list, toc_example)
        toc_rel_ans = []
        for ta, ra in zip(toc_ans, rel_ans):
            if ta == 1:
                toc_rel_ans.append(1)
            elif ta == 0 and ra != 0:
                toc_rel_ans.append(ra)
            else:
                toc_rel_ans.append(0)
        toc_rel_ans_tensor = torch.tensor(toc_rel_ans, dtype=torch.long, device=self.get_model_device())
        toc_rel_loss = self.logit_loss_forward(logit_tensor, toc_rel_ans_tensor)
        total_loss = toc_rel_loss

        # update node list
        max_index = max_index_tensor.detach().cpu().numpy().tolist()
        node_change_ids = self.get_keep_delete_up_down_node_ids(max_index, valid_list_ids['valid_list_count_to_valid_ids'])

        if self.gradient_accumulation_steps is not None:
            total_loss = total_loss / self.gradient_accumulation_steps

        res = {
            'total_loss': total_loss.detach().cpu().item(),
            'toc_rel_ans': toc_ans,
            'toc_rel_pred': max_index,
            'node_change_ids': node_change_ids,
        }

        return total_loss, res
