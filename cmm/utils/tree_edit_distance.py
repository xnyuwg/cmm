from cmm.tree_sim.apted.helpers import Tree
from cmm.tree_sim.apted.apted import APTED


class TreeEditDistanceManager:
    def __init__(self):
        pass

    @classmethod
    def get_tree_node_number_with_string(cls, tree):
        lb_count = 0
        rb_count = 0
        for c in tree:
            if c == '{':
                lb_count += 1
            if c == '}':
                rb_count += 1
        assert lb_count == rb_count
        return lb_count

    @classmethod
    def tree_edit_distance_with_string(cls, tree1, tree2):
        tree_node1, tree_node2 = map(Tree.from_text, [tree1, tree2])
        assert str(tree_node1) == tree1
        assert str(tree_node2) == tree2
        apted = APTED(tree_node1, tree_node2)
        ted = apted.compute_edit_distance()
        return ted

    @classmethod
    def convert_tree_to_string(cls, node, value_fn, children_fn):
        value = value_fn(node)
        children = children_fn(node)
        out = ['{', value]
        for child in children:
            out += cls.convert_tree_to_string(child, value_fn, children_fn)
        out += ['}']
        out = ''.join(out)
        return out

    @classmethod
    def tree_edit_distance_similarity_with_string(cls, tree1, tree2):
        ted = cls.tree_edit_distance_with_string(tree1=tree1, tree2=tree2)
        tree1_node = cls.get_tree_node_number_with_string(tree1)
        tree2_node = cls.get_tree_node_number_with_string(tree2)
        sim = max(0, 1 - ted / max(tree1_node, tree2_node))
        return sim
