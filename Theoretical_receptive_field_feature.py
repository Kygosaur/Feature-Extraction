import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Configure paths
INPUT_FOLDER = r"C:\Users\USER\OneDrive\Desktop\projects\Feature extraction\images"
OUTPUT_FOLDER = r"C:\Users\USER\OneDrive\Desktop\projects\Feature extraction\output"

class YOLOFeatureVisualizer:
    def __init__(self, model_name='yolov8m.pt'):
        self.model = YOLO(model_name)
        self.layer_info = {}  # Store information about each layer
        self.activation_history = {}  # Track activations across layers
        self.target_size = (640, 640)
        self.output_dir = Path(OUTPUT_FOLDER)
        self.output_dir.mkdir(exist_ok=True)

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
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

    def explain_layer(self, module, layer_name: str) -> Dict:
        """Generate beginner-friendly explanation of a layer's properties."""
        layer_info = {
            'name': layer_name,
            'type': module.__class__.__name__,
            'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
            'shape': None,  # Will be filled during forward pass
        }
        
        if isinstance(module, torch.nn.Conv2d):
            layer_info.update({
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
                'filters': module.out_channels,
                'description': f"Convolutional layer that applies {module.out_channels} filters "
                             f"of size {module.kernel_size} to detect visual patterns"
            })
        elif isinstance(module, torch.nn.BatchNorm2d):
            layer_info.update({
                'description': "Batch normalization layer that stabilizes and normalizes "
                             "the feature maps to help training"
            })
        elif isinstance(module, torch.nn.ReLU):
            layer_info.update({
                'description': "ReLU activation that introduces non-linearity by setting "
                             "negative values to zero"
            })
        elif isinstance(module, torch.nn.MaxPool2d):
            layer_info.update({
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'description': f"Max pooling layer that reduces spatial dimensions by "
                             f"selecting maximum values in {module.kernel_size}x{module.kernel_size} regions"
            })
            
        return layer_info

    def visualize_layer_progression(self):
        """Create a visual representation of how data flows through the model."""
        plt.figure(figsize=(15, 10))
        
        spatial_sizes = [info['shape'][2:] for info in self.layer_info.values() if info['shape'] is not None]
        layers = list(range(len(spatial_sizes)))
        
        plt.subplot(2, 1, 1)
        plt.plot(layers, [s[0] for s in spatial_sizes], 'b-', label='Height')
        plt.plot(layers, [s[1] for s in spatial_sizes], 'r--', label='Width')
        plt.title('Spatial Dimensions Through Layers')
        plt.xlabel('Layer Index')
        plt.ylabel('Dimension Size')
        plt.legend()
        plt.grid(True)
        
        channels = [info['shape'][1] for info in self.layer_info.values() if info['shape'] is not None]
        plt.subplot(2, 1, 2)
        plt.plot(layers, channels, 'g-')
        plt.title('Number of Channels Through Layers')
        plt.xlabel('Layer Index')
        plt.ylabel('Number of Channels')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_progression.png')
        plt.close()

    def visualize_receptive_field(self, layer_name: str):
        """Visualize the theoretical receptive field of a layer."""
        layer_idx = list(self.layer_info.keys()).index(layer_name)
        rf_size = 1
        for idx in range(layer_idx + 1):
            info = self.layer_info[list(self.layer_info.keys())[idx]]
            if isinstance(info.get('kernel_size'), (tuple, list, int)):
                kernel = info['kernel_size']
                if isinstance(kernel, int):
                    kernel = (kernel, kernel)
                rf_size += (kernel[0] - 1)
        
        plt.figure(figsize=(8, 8))
        img_size = 640
        center = img_size // 2
        rf_start = center - rf_size // 2
        rf_end = center + rf_size // 2
        
        img = np.zeros((img_size, img_size))
        img[rf_start:rf_end, rf_start:rf_end] = 1
        
        plt.imshow(img, cmap='gray')
        plt.title(f'Theoretical Receptive Field for {layer_name}\nSize: {rf_size}x{rf_size} pixels')
        plt.colorbar(label='Impact Strength')
        plt.savefig(self.output_dir / f'receptive_field_{layer_name}.png')
        plt.close()

    def generate_educational_report(self, feature_maps: Dict[str, torch.Tensor]):
        """Generate a beginner-friendly report explaining the feature extraction process."""
        report = []
        report.append("=== YOLO Feature Extraction Explained ===\n")
        
        report.append("How YOLO Processes Images:")
        report.append("1. The input image is resized to 640x640 pixels")
        report.append("2. The image passes through multiple layers that each detect different features")
        report.append("3. Early layers detect simple features (edges, colors)")
        report.append("4. Deeper layers detect complex features (shapes, objects)\n")
        
        report.append("=== Layer-by-Layer Analysis ===\n")
        for layer_name, info in self.layer_info.items():
            report.append(f"\nLayer: {layer_name}")
            report.append(f"Type: {info['type']}")
            report.append(f"Description: {info.get('description', 'Basic layer')}")
            
            if info['shape'] is not None:
                report.append(f"Output Shape: {info['shape']}")
                report.append(f"Number of Features: {info['shape'][1]}")
                report.append(f"Spatial Size: {info['shape'][2]}x{info['shape'][3]}")
            
            if 'kernel_size' in info:
                report.append(f"Kernel Size: {info['kernel_size']}")
                report.append(f"This means it looks at {info['kernel_size']}x{info['kernel_size']} pixel regions at a time")
            
            if layer_name in feature_maps:
                feat_map = feature_maps[layer_name]
                active_features = torch.mean((feat_map > 0).float()).item()
                report.append(f"Active Features: {active_features:.1%} of neurons are activated")
                max_activation = torch.max(feat_map).item()
                report.append(f"Strongest Activation: {max_activation:.2f}")
        
        report_path = self.output_dir / "yolo_educational_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
        print(f"Educational report saved to {report_path}")

    def hook_fn(self, module, input, output, layer_name: str):
        """Hook function that captures and explains layer information."""
        if layer_name in self.layer_info:
            self.layer_info[layer_name]['shape'] = output.shape
            
        with torch.no_grad():
            self.activation_history[layer_name] = {
                'mean_activation': torch.mean(output).item(),
                'max_activation': torch.max(output).item(),
                'active_features': torch.mean((output > 0).float()).item()
            }
        
        return output

    def extract_feature_maps(self, image_path: str) -> Dict[str, torch.Tensor]:
        """Extract feature maps with educational outputs."""
        print("\n=== Starting Feature Extraction Process ===")
        print("1. Loading and preparing image...")
        
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.resize_image(image)
        img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        print("2. Setting up layer monitoring...")
        feature_maps = {}
        hooks = []
        
        for name, module in self.model.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU)):
                layer_name = f"{name}_{module.__class__.__name__}"
                self.layer_info[layer_name] = self.explain_layer(module, layer_name)
                hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, ln=layer_name: self.hook_fn(m, i, o, ln)
                    )
                )
        
        print("3. Running image through the model...")
        try:
            with torch.no_grad():
                self.model.model(img_tensor)
            
            print("4. Generating educational visualizations...")
            self.visualize_layer_progression()
            
            for layer_name in self.layer_info:
                if isinstance(self.model.model.get_submodule(layer_name.split('_')[0]), torch.nn.Conv2d):
                    self.visualize_receptive_field(layer_name)
            
            print("5. Generating educational report...")
            self.generate_educational_report(feature_maps)
            
        finally:
            for hook in hooks:
                hook.remove()
        
        print("\n=== Feature Extraction Complete ===")
        print("Check the generated report and visualizations for detailed explanations!")
        
        return feature_maps

    def process_directory(self, directory_path: str):
        """Process all images in a directory."""
        dir_path = Path(directory_path)
        results = {}
        
        for image_path in dir_path.glob("*.*"):
            if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                print(f"\nProcessing {image_path.name}")
                try:
                    feature_maps = self.extract_feature_maps(str(image_path))
                    results[image_path.name] = {
                        'feature_maps': feature_maps,
                        'layer_info': self.layer_info.copy(),
                        'activation_history': self.activation_history.copy()
                    }
                except Exception as e:
                    print(f"Error processing {image_path.name}: {str(e)}")
        
        return results

def main():
    try:
        visualizer = YOLOFeatureVisualizer()
        
        # Process all images in the input folder
        print(f"Processing images from: {INPUT_FOLDER}")
        print(f"Saving outputs to: {OUTPUT_FOLDER}")
        
        results = visualizer.process_directory(INPUT_FOLDER)
        
        # Save overall results
        results_file = Path(OUTPUT_FOLDER) / "analysis_results.json"
        serializable_results = {
            img_name: {
                'layer_info': res['layer_info'],
                'activation_history': res['activation_history']
            }
            for img_name, res in results.items()
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        print(f"\nProcessing complete! Results saved to {OUTPUT_FOLDER}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()