from anytree import Node, PreOrderIter


class UtilStructure:
    def __init__(self):
        pass

    @staticmethod
    def find_max_and_number_index(x: list):
        max_number = max(x)
        index = x.index(max_number)
        return max_number, index

    @staticmethod
    def collect_rightest_node_of_tree(root_node, fun_to_find_child):
        children = fun_to_find_child(root_node)
        if len(children) == 0:
            return [root_node]
        right_child = children[-1]
        return [root_node] + UtilStructure.collect_rightest_node_of_tree(right_child, fun_to_find_child)

    @staticmethod
    def count_dict_update(dic, key, value=1):
        if key in dic:
            dic[key] += value
        else:
            dic[key] = value
        return dic

    @staticmethod
    def if_any_element_of_alist_in_blist(alist, blist):
        for a in alist:
            if a not in blist:
                return False
        return True

    @staticmethod
    def get_a_node_prev_next_siblings_in_tree(node):
        if node.parent is None:
            return None, None
        prev_node, next_node = None, None
        for i, child in enumerate(node.parent.children):
            if child.name == node.name:
                node_index = i
                break
        if node_index > 0:
            prev_node = node.parent.children[node_index - 1]
        if node_index < len(node.parent.children) - 1:
            next_node = node.parent.children[node_index + 1]
        return prev_node, next_node

    @staticmethod
    def copy_tree_with_root_node(node):
        new_node = Node(node.name)
        new_node.children = [UtilStructure.copy_tree_with_root_node(child) for child in node.children]
        return new_node

    @staticmethod
    def get_count_dict_sorted(count_dic, reverse=False):
        count_dic = [(k, v) for k, v in count_dic.items()]
        count_dic = sorted(count_dic, key=lambda x: x[1], reverse=reverse)
        return count_dic

    @staticmethod
    def str_to_bool(s):
        s = s.lower()
        if s in ['true', 't', 'yes', 'y', '1']:
            return True
        elif s in ['false', 'f', 'no', 'n', '0']:
            return False
        else:
            return None