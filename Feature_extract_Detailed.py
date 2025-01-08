import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple
import json
import logging
import datetime
import csv
from dataclasses import dataclass, asdict
import yaml
import os
from scipy.stats import skew, kurtosis

@dataclass
class AnalysisMetrics:
    """Data structure for storing analysis metrics"""
    image_name: str
    timestamp: str
    layer_name: str
    mean_activation: float
    max_activation: float
    activation_ratio: float
    complexity_score: float
    processing_time: float

class YOLOAnalysisLogger:
    def __init__(self, output_dir: str = "analysis_outputs"):
        """Initialize the logging system"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = self.output_dir / "logs"
        self.json_dir = self.output_dir / "json"
        self.csv_dir = self.output_dir / "csv"
        self.txt_dir = self.output_dir / "txt"
        
        for dir_path in [self.logs_dir, self.json_dir, self.csv_dir, self.txt_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_logging()
        
        self.results = {
            "session_info": {
                "session_id": self.session_id,
                "start_time": datetime.datetime.now().isoformat(),
                "processed_images": []
            },
            "analysis_results": {}
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.logs_dir / f"analysis_session_{self.session_id}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_analysis_start(self, image_name: str):
        """Log the start of analysis"""
        self.logger.info(f"Starting analysis for image: {image_name}")
        self.results["session_info"]["processed_images"].append({
            "image_name": image_name,
            "start_time": datetime.datetime.now().isoformat()
        })

    def log_layer_analysis(self, metrics: AnalysisMetrics):
        """Log layer analysis results"""
        if metrics.image_name not in self.results["analysis_results"]:
            self.results["analysis_results"][metrics.image_name] = {}
            
        self.results["analysis_results"][metrics.image_name][metrics.layer_name] = asdict(metrics)
        self._log_to_txt(metrics)
        self._log_to_csv(metrics)
        self.logger.info(f"Layer analysis completed - Image: {metrics.image_name}, Layer: {metrics.layer_name}")

    def _log_to_txt(self, metrics: AnalysisMetrics):
        """Log to text file"""
        txt_file = self.txt_dir / f"analysis_{metrics.image_name}.txt"
        with open(txt_file, 'a') as f:
            f.write(f"\n=== Layer Analysis: {metrics.layer_name} ===\n")
            for field, value in asdict(metrics).items():
                if isinstance(value, float):
                    f.write(f"{field}: {value:.4f}\n")
                else:
                    f.write(f"{field}: {value}\n")

    def _log_to_csv(self, metrics: AnalysisMetrics):
        """Log to CSV file"""
        csv_file = self.csv_dir / f"analysis_{self.session_id}.csv"
        file_exists = csv_file.exists()
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(metrics).keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(metrics))

    def log_error(self, image_name: str, error_msg: str):
        """Log error messages"""
        self.logger.error(f"Error processing {image_name}: {error_msg}")
        if self.results["session_info"]["processed_images"]:
            self.results["session_info"]["processed_images"][-1]["error"] = error_msg

    def save_final_results(self):
        """Save final results in multiple formats"""
        self.results["session_info"]["end_time"] = datetime.datetime.now().isoformat()
        
        # Save JSON
        json_path = self.json_dir / f"analysis_results_{self.session_id}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Save YAML
        yaml_path = self.output_dir / f"analysis_summary_{self.session_id}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)
            
        self._generate_summary_report()
        self.logger.info(f"Analysis session {self.session_id} completed")

    def _generate_summary_report(self):
        """Generate summary report"""
        report_path = self.output_dir / f"analysis_report_{self.session_id}.txt"
        with open(report_path, 'w') as f:
            f.write("=== YOLO Feature Analysis Summary Report ===\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {self.results['session_info']['start_time']}\n")
            f.write(f"End Time: {self.results['session_info']['end_time']}\n")
            f.write(f"Total Images: {len(self.results['session_info']['processed_images'])}\n\n")
            
            for image_name, image_results in self.results["analysis_results"].items():
                f.write(f"\nImage: {image_name}\n")
                for layer_name, metrics in image_results.items():
                    f.write(f"  {layer_name}:\n")
                    f.write(f"    Mean Activation: {metrics['mean_activation']:.4f}\n")
                    f.write(f"    Complexity Score: {metrics['complexity_score']:.4f}\n")

class CompleteYOLOAnalyzer:
    def __init__(self, model_name='yolov8m.pt', output_dir: str = "analysis_outputs"):
        """Initialize the analyzer with both analysis and logging capabilities"""
        self.model = YOLO(model_name)
        self.logger = YOLOAnalysisLogger(output_dir)
        self.processed_images = []
        self.failed_images = []

    def analyze_feature_map(self, feature_map: torch.Tensor) -> Dict:
        """Analyze a single feature map"""
        feat_np = feature_map.cpu().numpy()
        
        # Basic statistics
        stats = {
            'mean_activation': float(np.mean(feat_np)),
            'max_activation': float(np.max(feat_np)),
            'min_activation': float(np.min(feat_np)),
            'std_activation': float(np.std(feat_np)),
            'activation_ratio': float(np.sum(feat_np > 0) / feat_np.size),
            'skewness': float(skew(feat_np.flatten())),
            'kurtosis': float(kurtosis(feat_np.flatten()))
        }
        
        # Spatial analysis
        spatial_stats = {
            'spatial_variance': float(np.var(feat_np, axis=(2, 3)).mean()),
            'channel_correlation': float(np.corrcoef(feat_np.reshape(feat_np.shape[1], -1)).mean())
        }
        
        # Complexity analysis
        gradients_y = np.diff(feat_np, axis=2)
        gradients_x = np.diff(feat_np, axis=3)
        complexity_score = float(np.mean(np.abs(gradients_x)) + np.mean(np.abs(gradients_y)))
        
        return {
            'basic_stats': stats,
            'spatial_stats': spatial_stats,
            'complexity_score': complexity_score
        }

    def visualize_feature_map(self, feature_map: torch.Tensor, layer_name: str, image_name: str):
        """Visualize feature maps"""
        feature_map = feature_map[0].cpu().numpy()
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
        save_path = self.logger.output_dir / f"feature_maps_{image_name}_{layer_name}.png"
        plt.savefig(save_path)
        plt.close()

    def extract_feature_maps(self, image_path: str) -> Dict:
        """Extract feature maps from an image"""
        image_name = Path(image_path).name
        self.logger.log_analysis_start(image_name)
        
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            feature_maps = {}
            layer_counts = {}
            
            def hook_fn(module, input, output, name):
                layer_type = module.__class__.__name__
                if layer_type not in layer_counts:
                    layer_counts[layer_type] = 0
                layer_counts[layer_type] += 1
                
                layer_name = f"{layer_type}_{layer_counts[layer_type]}"
                feature_maps[layer_name] = output

            hooks = []
            for name, module in self.model.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    hooks.append(module.register_forward_hook(
                        lambda module, input, output, name=name: hook_fn(module, input, output, name)
                    ))
            
            try:
                with torch.no_grad():
                    self.model.model(img_tensor)
                
                # Process each feature map
                for layer_name, feature_map in feature_maps.items():
                    start_time = datetime.datetime.now()
                    
                    # Analyze and visualize
                    analysis = self.analyze_feature_map(feature_map)
                    self.visualize_feature_map(feature_map, layer_name, image_name)
                    
                    # Log results
                    metrics = AnalysisMetrics(
                        image_name=image_name,
                        timestamp=datetime.datetime.now().isoformat(),
                        layer_name=layer_name,
                        mean_activation=analysis['basic_stats']['mean_activation'],
                        max_activation=analysis['basic_stats']['max_activation'],
                        activation_ratio=analysis['basic_stats']['activation_ratio'],
                        complexity_score=analysis['complexity_score'],
                        processing_time=(datetime.datetime.now() - start_time).total_seconds()
                    )
                    self.logger.log_layer_analysis(metrics)
                
                self.processed_images.append(image_name)
                return feature_maps
                
            finally:
                for hook in hooks:
                    hook.remove()
                    
        except Exception as e:
            self.failed_images.append((image_name, str(e)))
            self.logger.log_error(image_name, str(e))
            return None

    def process_images(self, folder_path: str):
        """Process all images in a folder"""
        base_path = Path(folder_path)
        if not base_path.exists():
            raise ValueError(f"Folder not found: {base_path}")
        
        image_files = sorted(base_path.glob('*.png'))
        results = {}
        
        for img_path in image_files:
            feature_maps = self.extract_feature_maps(str(img_path))
            if feature_maps is not None:
                results[img_path.name] = feature_maps
        
        self.logger.save_final_results()
        return results

def main():
    """Main function to run the analysis"""
    folder_path = r"C:\Users\USER\OneDrive\Desktop\projects\al-Andalusia\images"
    output_dir = "yolo_analysis_results"
    
    try:
        analyzer = CompleteYOLOAnalyzer(output_dir=output_dir)
        results = analyzer.process_images(folder_path)
        
        print(f"\nAnalysis complete. Results saved to {output_dir}")
        print("Available output files:")
        print(f"- Detailed logs: {output_dir}/logs/")
        print(f"- JSON results: {output_dir}/json/")
        print(f"- CSV data: {output_dir}/csv/")
        print(f"- Text reports: {output_dir}/txt/")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()