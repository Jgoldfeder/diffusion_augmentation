import random

class AugmentationNode:
    def __init__(self, augmentation_type=None):
        self.augmentation_type = augmentation_type
        self.left_child_probability = 0.0
        self.right_child_probability = 0.0
        self.left = None
        self.right = None

augmentation_types = ['canny', 'depth', 'seg', 'color', 'nerf', 'classical', 'none']

def print_tree(node, level=0, direction='root'):
    if node:
        if not node.left and not node.right:
            # Leaf nodes: only show augmentation type
            edge_info = f"(Augmentation: {node.augmentation_type})"
        else:
            # Non-leaf nodes: show augmentation type and probabilities
            edge_info = f"(Augmentation: {node.augmentation_type}, L_prob: {node.left_child_probability:.2f}, R_prob: {node.right_child_probability:.2f})"
        print('  ' * level + f"{direction}: {edge_info}")
        if node.left:
            print_tree(node.left, level + 1, 'L')
        if node.right:
            print_tree(node.right, level + 1, 'R')

def initialize_augmentation_tree(depth=3):
    def create_node(current_depth, augmentation_type=None, is_root=False):
        if current_depth == 0:
            return None
        
        node = AugmentationNode(augmentation_type=augmentation_type)
        
        # Only create children if depth > 1
        if current_depth > 1:
            # Create left and right children
            node.left = create_node(current_depth - 1, augmentation_type)
            node.right = create_node(current_depth - 1, augmentation_type)
            
            # Set augmentation type for non-leaf nodes
            node.augmentation_type = random.choice(augmentation_types)
        else:
            # This is a leaf node - set augmentation type but no children
            node.augmentation_type = random.choice(augmentation_types)
            node.left = None
            node.right = None
        
        # Distribute probabilities between children
        if node.left and node.right:
            # Randomly assign a portion between 0.3 and 0.7 for left child
            left_ratio = random.uniform(0.3, 0.7)  # This ensures more balanced splits
            right_ratio = 1.0 - left_ratio
            
            node.left_child_probability = left_ratio
            node.right_child_probability = right_ratio
        else:
            # Leaf nodes have no children, so probabilities are 0
            node.left_child_probability = 0.0
            node.right_child_probability = 0.0
            
        return node
    
    # Create root node with is_root=True
    root = create_node(depth, is_root=True)
    
    print("Augmentation Tree Structure:")
    print_tree(root)
    print("--------------------------------")
    return root

def test_augmentation_tree():
    depth = 3
    print(f"Testing Augmentation Tree with depth {depth}...")
    root = initialize_augmentation_tree(depth)

if __name__ == "__main__":
    test_augmentation_tree()