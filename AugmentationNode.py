import random

class AugmentationNode:
    def __init__(self, parent_edge_type=None, parent_edge_probability=0.5):
        self.parent_edge_type = parent_edge_type
        self.parent_edge_probability = parent_edge_probability
        self.left = None
        self.right = None

def initialize_augmentation_tree(depth=3):
    augmentation_types = ['canny', 'depth', 'seg', 'color', 'nerf', 'classical']
    
    def create_node(current_depth, parent_edge_type=None):
        if current_depth == 0:
            return None
        
        if current_depth < depth:
            edge_type = random.choice(augmentation_types)
            total_prob = random.uniform(0.3, 0.7)
            left_prob = total_prob
            right_prob = 1.0 - total_prob
        else:
            edge_type = None
            left_prob = right_prob = 0.0
        
        node = AugmentationNode(parent_edge_type=edge_type, parent_edge_probability=left_prob)
        node.left = create_node(current_depth - 1, edge_type)
        node.right = create_node(current_depth - 1, edge_type)
        
        if node.left and node.right:
            node.left.parent_edge_probability = left_prob
            node.right.parent_edge_probability = right_prob
        
        return node
    
    root = create_node(depth)
    
    def print_tree(node, level=0, direction='root'):
        if node:
            edge_info = f"(edge: {node.parent_edge_type}, prob: {node.parent_edge_probability:.2f})" if node.parent_edge_type else "(root)"
            print('  ' * level + f"{direction}: {edge_info}")
            if node.left:
                print_tree(node.left, level + 1, 'L')
            if node.right:
                print_tree(node.right, level + 1, 'R')
    
    print("Augmentation Tree Structure:")
    print_tree(root)
    
    return root

def test_augmentation_tree():
    depth = 4
    print(f"Testing Augmentation Tree with depth {depth}...")
    root = initialize_augmentation_tree(depth)

if __name__ == "__main__":
    test_augmentation_tree()