import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import wandb
import pygad
import logging
import argparse
import torch
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Dimension reduction and clustering libraries
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.manifold import TSNE

import network_model
import dataset_manager
from network_model import ModelResults, ModelType
from dataset_manager import FolderDataset
from augmentation_tree import BinaryAugmentationNode, AugmentationType, TreeAugmentedDataset, TreeAugmentedDatasetFromDataset, ProbabilityLimits

from torch.utils.data import DataLoader
from torch import nn
import torchvision
from sklearn.metrics import silhouette_score

from genetic_algorithm import genome_to_number, genome_to_tree, GAHelper

import timm
from transformers import AutoImageProcessor, Dinov2ForImageClassification

try:
    import clip
except ImportError:
    clip = None

class ClusteringVisualizer(GAHelper):
      def visualize(self, image_encoder='vit224', dimension_reduction='umap'):
        genome = input("Enter the genome: ")
        genome = [float(x) for x in genome.split(',')]
        genome_number = genome_to_number(genome)
        if genome_number in self.fitness_cache:
            logging.info('fitness function call using cached fitness')
            return self.fitness_cache[genome_number]
        self.tree_evals_per_generation[-1] += 1
        
        node = genome_to_tree(genome)
        dataset = FolderDataset(self.train_path)
        tree_augmented_train_dataset = TreeAugmentedDatasetFromDataset(dataset, node, self.num_augmentations_per_image)
        
        data_loader = DataLoader(tree_augmented_train_dataset, batch_size=256, shuffle=False, num_workers=2)
        if image_encoder == 'vit224':
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            if hasattr(model, "head"):
                model.head = nn.Identity()
            elif hasattr(model, "classifier"):
                model.classifier = nn.Identity()
        elif image_encoder == "clip":
            if clip is None:
                raise ImportError("CLIP is not installed. Please install it.")
            # Load the ViT-B/32 variant of CLIP.
            model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        elif image_encoder == "resnet50":
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Identity()
        elif image_encoder == "vit":
            # Use torchvision's ViT-B/16
            model = torchvision.models.vit_b_16(pretrained=True)
            # Replace the classification head with an identity.
            if hasattr(model, "heads"):
                model.heads = nn.Identity()
            else:
                model.fc = nn.Identity()
        elif image_encoder == "dino":
            # Load DINOv2 from the transformers library.
            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
            model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
            model.eval()
        else:
            raise ValueError(f"Unsupported image encoder: {image_encoder}")
        
        embeddings = []
        true_labels = []
        
        with torch.no_grad():
            for (images, labels) in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                features = model(images)
                embeddings.append(features.cpu().numpy())
                true_labels.append(labels.cpu().numpy())
                
        embeddings = np.concatenate(embeddings, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        if dimension_reduction == "umap":
            umap_embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            classes = np.unique(true_labels)

            for cls in classes:
                idx = np.where(true_labels == cls)
                plt.scatter(
                    umap_embedding[idx, 0],
                    umap_embedding[idx, 1],
                    s=5,
                    label=str(cls)
                )

            plt.legend(title="Class Label")
            plt.title('UMAP projection')
            plt.xlabel('UMAP-1')
            plt.ylabel('UMAP-2')
            plt.tight_layout()
            plt.savefig("umap_projection.png")  # Save the plot as an image file
        elif dimension_reduction == "tsne":
            if len(true_labels) > 50:
                perplexity = 50
            else:
                perplexity = len(true_labels) - 1
            tsne_embedding = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            classes = np.unique(true_labels)

            for cls in classes:
                idx = np.where(true_labels == cls)
                plt.scatter(tsne_embedding[idx, 0], tsne_embedding[idx, 1], s=5, label=str(cls))
            plt.legend(title="Class Label")
            plt.title("t-SNE projection")
            plt.xlabel("t-SNE-1")
            plt.ylabel("t-SNE-2")
            plt.tight_layout()
            plt.savefig("tsne_projection.png")
        elif dimension_reduction == "umap3d":
            umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(embeddings)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            classes = np.unique(true_labels)
            for cls in classes:
                idx = np.where(true_labels == cls)
                ax.scatter(
                    umap_3d[idx, 0],
                    umap_3d[idx, 1],
                    umap_3d[idx, 2],
                    label=str(cls),
                    s=5
                )

            ax.set_title('3D UMAP Projection')
            ax.set_xlabel('UMAP-1')
            ax.set_ylabel('UMAP-2')
            ax.set_zlabel('UMAP-3')
            ax.legend(title="Class Label")
            plt.tight_layout()
            plt.savefig("umap3d_projection.png")
        elif dimension_reduction == "tsne3d":
            if len(true_labels) > 50:
                perplexity = 50
            else:
                perplexity = len(true_labels) - 1 
            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity).fit_transform(embeddings)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for cls in np.unique(true_labels):
                idx = np.where(true_labels == cls)
                ax.scatter(
                    tsne_3d[idx, 0],
                    tsne_3d[idx, 1],
                    tsne_3d[idx, 2],
                    label=str(cls),
                    s=5
                )

            ax.set_title('3D t-SNE Projection')
            ax.set_xlabel('t-SNE-1')
            ax.set_ylabel('t-SNE-2')
            ax.set_zlabel('t-SNE-3')
            ax.legend(title="Class Label")
            plt.tight_layout()
            plt.savefig("tsne3d_projection.png")


def parse_args():
    parser = argparse.ArgumentParser(description='Run genetic algorithm for augmentation tree optimization')
    parser.add_argument('--num_generations', type=int, default=3, help='Number of generations')
    parser.add_argument('--sol_per_pop', type=int, default=10, help='Solutions per population')
    parser.add_argument('--num_parents_mating', type=int, default=4, help='Number of parents for mating')
    parser.add_argument('--keep_elitism', type=int, default=1, help='Number of elites to keep')
    parser.add_argument('--keep_parents', type=int, default=4, help='Number of parents to keep')
    parser.add_argument('--mutation_percent', type=int, default=10, help='Mutation percentage')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--num_ways', type=int, required=True, help='Number of ways (classes)')
    parser.add_argument('--num_shots', type=int, default=1, help='Number of shots (examples per class)')
    parser.add_argument('--subset', type=int, required=True, help='Which subset of classes to use')
    parser.add_argument('--model_type', type=str, default='resnet50', help='Which base model to use')
    parser.add_argument('--tree_depth', type=int, required=True, help='Depth of the augmentation tree')
    parser.add_argument('--num_augmentations_per_image', type=int, default=5, help='Number of augmentations per image to expand dataset by')
    parser.add_argument('--num_iterations_for_val', type=int, default=20, help='Number of iterations to train each tree before getting loss from val')
    parser.add_argument('--num_iterations_for_test', type=int, default=400, help='Number of iterations to train best tree before final testing')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--one_shot_training_loss', type=bool, default=False, help='Whether to use one shot training loss')
    parser.add_argument('--one_shot_clustering', type=bool, default=False, help='Whether to use one shot clustering')
    parser.add_argument('--image_encoder', type=str, default='vit224', help='Which image encoder to use')
    parser.add_argument('--dimension_reduction', type=str, default='umap', help='Which dimension reduction method to use')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    # might also need to seed numpy here too
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    clustering_visualizer = ClusteringVisualizer(
        dataset_name=args.dataset,
        num_ways=args.num_ways,
        num_shots=args.num_shots,
        subset=args.subset,
        model_type=ModelType(args.model_type),
        tree_depth=args.tree_depth,
        num_augmentations_per_image=args.num_augmentations_per_image,
        num_iterations_for_val=args.num_iterations_for_val,
        num_iterations_for_test=args.num_iterations_for_test,
        device='cpu'
    )

    print('hi')	
    clustering_visualizer.visualize(args.image_encoder, args.dimension_reduction)