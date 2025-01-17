import random

class AugmentationNode:
    def __init__(self, parent_edge_type=None):
        self.parent_edge_type = parent_edge_type
        self.left_child_probability = 0.0
        self.right_child_probability = 0.0
        self.left = None
        self.right = None

augmentation_types = ['canny', 'depth', 'seg', 'color', 'nerf', 'classical', 'none']

def print_tree(node, level=0, direction='root'):
        if node:
            if direction == 'root':
                edge_info = f"(root, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
            else:
                edge_info = f"(edge: {node.parent_edge_type}, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
            print('  ' * level + f"{direction}: {edge_info}")
            if node.left:
                print_tree(node.left, level + 1, 'L')
            if node.right:
                print_tree(node.right, level + 1, 'R')

def initialize_augmentation_tree(depth=3):
    def create_node(current_depth, parent_edge_type=None, is_root=False):
        if current_depth == 0:
            return None
        
        # For non-leaf nodes, randomly select augmentation type
        if current_depth < depth:
            edge_type = random.choice(augmentation_types)
        else:
            edge_type = None
        
        node = AugmentationNode(parent_edge_type=edge_type)
        
        # Create left and right children
        node.left = create_node(current_depth - 1, edge_type)
        node.right = create_node(current_depth - 1, edge_type)
        
        # Distribute probabilities between children
        if node.left and node.right:
            # Randomly assign a portion between 0.3 and 0.7 for left child
            left_ratio = random.uniform(0.3, 0.7)  # This ensures more balanced splits
            right_ratio = 1.0 - left_ratio
            
            node.left_child_probability = left_ratio
            node.right_child_probability = right_ratio
        elif node.left:
            node.left_child_probability = 1.0
            node.right_child_probability = 0.0
        elif node.right:
            node.left_child_probability = 0.0
            node.right_child_probability = 1.0
            
        return node
    
    # Create root node with is_root=True
    root = create_node(depth, is_root=True)
    
    print("Augmentation Tree Structure:")
    print_tree(root)
    print("--------------------------------")
    return root

def test_augmentation_tree():
    depth = 4
    print(f"Testing Augmentation Tree with depth {depth-1}...")
    root = initialize_augmentation_tree(depth)

if __name__ == "__main__":
    test_augmentation_tree()