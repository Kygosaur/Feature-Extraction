import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple

"""
YOLOFeatureVisualizer: A class for extracting and visualizing feature maps from YOLOv8 models.
This tool helps understand how the YOLO model processes images at different layers by:
1. Extracting intermediate feature maps from convolutional layers
2. Generating visualizations of these feature maps
3. Computing statistical analyses of the activations
"""

class YOLOFeatureVisualizer:
    def __init__(self, model_name='yolov8m.pt'):  
        """
        Initialize the YOLO model and tracking variables
        Args:
            model_name: Path to YOLO model weights (defaults to YOLOv8 medium)
        """
        self.model = YOLO(model_name)  # Load YOLO model
        self.processed_images = []      # Track successfully processed images
        self.failed_images = []         # Track failed images and their error messages
        self.target_size = (640, 640)   # YOLO's expected input size
        
    def print_feature_statistics(self, feature_maps: Dict[str, torch.Tensor], image_name: str):
        """
        Analyze and print statistical information about feature maps.
        This helps understand the activation patterns at each layer.
        
        For each feature map, calculates and prints:
        - Basic shape information
        - Overall statistics (mean, std, min, max)
        - Per-channel statistics for first 5 channels
        
        Args:
            feature_maps: Dictionary mapping layer names to their feature tensors
            image_name: Name of the image being processed
        """
        print(f"\n=== Feature Map Statistics for {image_name} ===")
        
        for layer_name, feature_map in feature_maps.items():
            # Convert to numpy for statistical calculations
            feat_np = feature_map.cpu().numpy()
            
            # Print basic information about the layer
            print(f"\nLayer: {layer_name}")
            print(f"Shape: {feat_np.shape}")
            print(f"Number of channels: {feat_np.shape[1]}")
            print(f"Spatial dimensions: {feat_np.shape[2]}x{feat_np.shape[3]}")
            
            # Calculate and print overall statistics
            print("\nNumerical Statistics:")
            print(f"Mean: {np.mean(feat_np):.4f}")
            print(f"Std: {np.std(feat_np):.4f}")
            print(f"Min: {np.min(feat_np):.4f}")
            print(f"Max: {np.max(feat_np):.4f}")
            
            # Analyze individual channels
            print("\nChannel Statistics (first 5 channels):")
            for i in range(min(5, feat_np.shape[1])):
                channel_data = feat_np[0, i]
                print(f"Channel {i}:")
                print(f"  Mean: {np.mean(channel_data):.4f}")
                print(f"  Max Activation: {np.max(channel_data):.4f}")
                print(f"  Active Cells: {np.sum(channel_data > 0)}")  # Count non-zero activations

    def visualize_feature_map(self, feature_map: torch.Tensor, layer_name: str, image_name: str):
        """
        Create and save visualizations of feature maps.
        Generates a 4x4 grid showing up to 16 channels from the feature map.
        
        Args:
            feature_map: Tensor containing the feature maps for a layer
            layer_name: Name of the layer being visualized
            image_name: Name of the input image
        """
        # Convert to numpy and select first image if batch size > 1
        feature_map = feature_map[0].cpu().numpy()
        
        # Set up the visualization grid
        num_channels = min(16, feature_map.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        fig.suptitle(f'Feature Maps for {layer_name} - {image_name}')
        
        # Plot each channel's feature map
        for idx in range(16):
            if idx < num_channels:
                ax = axes[idx//4, idx%4]
                # Use viridis colormap for better visualization of activations
                im = ax.imshow(feature_map[idx], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Channel {idx}')
            else:
                axes[idx//4, idx%4].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = Path(f"feature_maps_{image_name.replace('.png', '')}_{layer_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved feature map visualization to {save_path}")

    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize and pad images to YOLO's expected input size while maintaining aspect ratio.
        This prevents distortion while ensuring the image fits YOLO's requirements.
        
        Args:
            image: Input PIL Image
        Returns:
            Resized and padded PIL Image
        """
        current_width, current_height = image.size
        
        # Skip if already correct size
        if (current_width, current_height) == self.target_size:
            return image
            
        # Calculate aspect ratios
        target_aspect = self.target_size[0] / self.target_size[1]
        current_aspect = current_width / current_height
        
        # Resize maintaining aspect ratio
        if current_aspect > target_aspect:
            new_width = self.target_size[0]
            new_height = int(new_width / current_aspect)
        else:
            new_height = self.target_size[1]
            new_width = int(new_height * current_aspect)
            
        # Use LANCZOS resampling for high-quality resizing
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create black padding
        padded_image = Image.new('RGB', self.target_size, (0, 0, 0))
        
        # Center the image
        left_padding = (self.target_size[0] - new_width) // 2
        top_padding = (self.target_size[1] - new_height) // 2
        
        padded_image.paste(resized_image, (left_padding, top_padding))
        
        return padded_image

    def extract_feature_maps(self, image_path: str) -> Dict[str, torch.Tensor]:
        """
        Process a single image to extract feature maps from YOLO's convolutional layers.
        
        Workflow:
        1. Load and preprocess the image
        2. Set up hooks to capture intermediate layer outputs
        3. Run the image through YOLO
        4. Generate visualizations and statistics
        
        Args:
            image_path: Path to the input image
        Returns:
            Dictionary mapping layer names to their feature maps
        """
        image_name = Path(image_path).name
        print(f"\n=== Processing {image_name} ===")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            original_size = image.size
            print(f"Original image size: {original_size}")
            
            image = self.resize_image(image)
            print(f"Processed image size: {image.size}")
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            
            # Storage for feature maps and layer counting
            feature_maps = {}
            layer_counts = {}
            
            # Hook function to capture layer outputs
            def hook_fn(module, input, output):
                layer_type = module.__class__.__name__
                if layer_type not in layer_counts:
                    layer_counts[layer_type] = 0
                layer_counts[layer_type] += 1
                layer_name = f"{layer_type}_{layer_counts[layer_type]}"
                feature_maps[layer_name] = output
            
            # Register hooks for convolutional layers
            hooks = []
            for name, module in self.model.model.named_modules():
                # Only capture first three model blocks
                if isinstance(module, torch.nn.Conv2d) and any(x in name for x in ['model.0', 'model.1', 'model.2']):
                    hooks.append(module.register_forward_hook(hook_fn))
            
            try:
                # Run inference
                with torch.no_grad():
                    self.model.model(img_tensor.to(self.model.device))
                self.processed_images.append(image_name)
                
                # Generate visualizations and statistics
                self.print_feature_statistics(feature_maps, image_name)
                for layer_name, feature_map in feature_maps.items():
                    self.visualize_feature_map(feature_map, layer_name, image_name)
                    
            except Exception as e:
                self.failed_images.append((image_name, str(e)))
                print(f"Failed to process image: {str(e)}")
                return None
            finally:
                # Clean up hooks
                for hook in hooks:
                    hook.remove()
            
            return feature_maps
            
        except Exception as e:
            self.failed_images.append((image_name, str(e)))
            print(f"Error loading image: {str(e)}")
            return None

    def process_al_andalusia(self, full_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Batch process all images in a specified directory.
        
        Args:
            full_path: Path to directory containing images
        Returns:
            Dictionary mapping image names to their feature maps
        """
        base_path = Path(full_path)
        
        if not base_path.exists():
            raise ValueError(f"Folder not found: {base_path}")
        
        print(f"\n=== Processing images with YOLOv8m in folder: {base_path} ===")
        
        # Find all image files
        image_files = sorted([
            f for ext in ['*.png', '*.jpg', '*.jpeg'] 
            for f in base_path.glob(ext)
        ])
        
        if not image_files:
            print(f"No PNG images found in {base_path}")
            return {}
        
        # Print summary of found images
        print(f"\nFound {len(image_files)} images to process:")
        for img_path in image_files:
            print(f"- {img_path.name}")
        
        # Process each image
        results = {}
        for img_path in image_files:
            print(f"\nProcessing: {img_path.name}")
            feature_maps = self.extract_feature_maps(str(img_path))
            if feature_maps is not None:
                results[img_path.name] = feature_maps
        
        # Print processing summary
        print("\n=== Processing Summary ===")
        print(f"Total images found: {len(image_files)}")
        print(f"Successfully processed: {len(self.processed_images)}")
        print(f"Failed to process: {len(self.failed_images)}")
        
        if self.failed_images:
            print("\nFailed images and reasons:")
            for img, reason in self.failed_images:
                print(f"- {img}: {reason}")
        
        return results

def main():
    """
    Main entry point for the script.
    Processes images from a specified folder and generates visualizations.
    """
    folder_path = r"C:\Users\USER\OneDrive\Desktop\projects\Feature extraction\images"
    
    try:
        visualizer = YOLOFeatureVisualizer()  # Uses YOLOv8m by default
        results = visualizer.process_al_andalusia(folder_path)
        
        print(f"\nCompleted processing with {len(results)} successful extractions")
        print("Feature map visualizations have been saved to individual files")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()