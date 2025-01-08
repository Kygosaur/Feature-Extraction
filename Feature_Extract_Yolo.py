import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple

class YOLOFeatureVisualizer:
    def __init__(self, model_name='yolov8m.pt'):  
        self.model = YOLO(model_name)
        self.processed_images = []
        self.failed_images = []
        self.target_size = (640, 640)  # Define target size as class attribute
        
    def print_feature_statistics(self, feature_maps: Dict[str, torch.Tensor], image_name: str):
        """
        Print detailed statistics for each feature map
        """
        print(f"\n=== Feature Map Statistics for {image_name} ===")
        
        for layer_name, feature_map in feature_maps.items():
            feat_np = feature_map.cpu().numpy()
            
            print(f"\nLayer: {layer_name}")
            print(f"Shape: {feat_np.shape}")
            print(f"Number of channels: {feat_np.shape[1]}")
            print(f"Spatial dimensions: {feat_np.shape[2]}x{feat_np.shape[3]}")
            print("\nNumerical Statistics:")
            print(f"Mean: {np.mean(feat_np):.4f}")
            print(f"Std: {np.std(feat_np):.4f}")
            print(f"Min: {np.min(feat_np):.4f}")
            print(f"Max: {np.max(feat_np):.4f}")
            
            print("\nChannel Statistics (first 5 channels):")
            for i in range(min(5, feat_np.shape[1])):
                channel_data = feat_np[0, i]
                print(f"Channel {i}:")
                print(f"  Mean: {np.mean(channel_data):.4f}")
                print(f"  Max Activation: {np.max(channel_data):.4f}")
                print(f"  Active Cells: {np.sum(channel_data > 0)}")

    def visualize_feature_map(self, feature_map: torch.Tensor, layer_name: str, image_name: str):
        """
        Visualize feature maps for a specific layer
        """
        # Convert to numpy and take the first image if batch size > 1
        feature_map = feature_map[0].cpu().numpy()
        
        # Create a figure to show first 16 channels (or less if fewer channels)
        num_channels = min(16, feature_map.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        fig.suptitle(f'Feature Maps for {layer_name} - {image_name}')
        
        for idx in range(16):
            if idx < num_channels:
                ax = axes[idx//4, idx%4]
                im = ax.imshow(feature_map[idx], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Channel {idx}')
            else:
                axes[idx//4, idx%4].axis('off')
        
        plt.tight_layout()
        
        save_path = Path(f"feature_maps_{image_name.replace('.png', '')}_{layer_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved feature map visualization to {save_path}")

    def resize_image(self, image: Image.Image) -> Image.Image:

        current_width, current_height = image.size
        
        if (current_width, current_height) == self.target_size:
            return image
            
        target_aspect = self.target_size[0] / self.target_size[1]
        current_aspect = current_width / current_height
        
        if current_aspect > target_aspect:
            new_width = self.target_size[0]
            new_height = int(new_width / current_aspect)
        else:
            new_height = self.target_size[1]
            new_width = int(new_height * current_aspect)
            
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        padded_image = Image.new('RGB', self.target_size, (0, 0, 0))
        
        left_padding = (self.target_size[0] - new_width) // 2
        top_padding = (self.target_size[1] - new_height) // 2
        
        padded_image.paste(resized_image, (left_padding, top_padding))
        
        return padded_image

    def extract_feature_maps(self, image_path: str) -> Dict[str, torch.Tensor]:
        """
        Extract and visualize feature maps from a single image.
        """
        image_name = Path(image_path).name
        print(f"\n=== Processing {image_name} ===")
        
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            original_size = image.size
            print(f"Original image size: {original_size}")
            
            image = self.resize_image(image)
            print(f"Processed image size: {image.size}")
            
            img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            feature_maps = {}
            layer_counts = {}
            
            def hook_fn(module, input, output):
                layer_type = module.__class__.__name__
                if layer_type not in layer_counts:
                    layer_counts[layer_type] = 0
                layer_counts[layer_type] += 1
                layer_name = f"{layer_type}_{layer_counts[layer_type]}"
                feature_maps[layer_name] = output
            
            hooks = []
            for name, module in self.model.model.named_modules():
                if isinstance(module, torch.nn.Conv2d) and any(x in name for x in ['model.0', 'model.1', 'model.2']):
                    hooks.append(module.register_forward_hook(hook_fn))
            
            try:
                with torch.no_grad():
                    self.model.model(img_tensor.to(self.model.device))
                self.processed_images.append(image_name)
                
                self.print_feature_statistics(feature_maps, image_name)
                for layer_name, feature_map in feature_maps.items():
                    self.visualize_feature_map(feature_map, layer_name, image_name)
                    
            except Exception as e:
                self.failed_images.append((image_name, str(e)))
                print(f"Failed to process image: {str(e)}")
                return None
            finally:
                for hook in hooks:
                    hook.remove()
            
            return feature_maps
            
        except Exception as e:
            self.failed_images.append((image_name, str(e)))
            print(f"Error loading image: {str(e)}")
            return None

    def process_al_andalusia(self, full_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        base_path = Path(full_path)
        
        if not base_path.exists():
            raise ValueError(f"Folder not found: {base_path}")
        
        print(f"\n=== Processing images with YOLOv8m in folder: {base_path} ===")
        
        image_files = sorted([
            f for ext in ['*.png', '*.jpg', '*.jpeg'] 
            for f in base_path.glob(ext)
        ])
        
        if not image_files:
            print(f"No PNG images found in {base_path}")
            return {}
        
        print(f"\nFound {len(image_files)} images to process:")
        for img_path in image_files:
            print(f"- {img_path.name}")
        
        results = {}
        for img_path in image_files:
            print(f"\nProcessing: {img_path.name}")
            feature_maps = self.extract_feature_maps(str(img_path))
            if feature_maps is not None:
                results[img_path.name] = feature_maps
        
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
    folder_path = r"C:\Users\USER\OneDrive\Desktop\projects\al-Andalusia\images"
    
    try:
        visualizer = YOLOFeatureVisualizer()  # Uses YOLOv8m by default
        results = visualizer.process_al_andalusia(folder_path)
        
        print(f"\nCompleted processing with {len(results)} successful extractions")
        print("Feature map visualizations have been saved to individual files")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()