# YOLO Feature Visualizer

A Python tool for extracting and visualizing feature maps from YOLOv8 models. This tool helps researchers and developers understand the internal representations learned by YOLO models by visualizing intermediate feature maps and providing detailed statistical analysis.

## Features

- Extract feature maps from YOLOv8 models
- Generate visualizations of feature maps for each layer
- Provide detailed statistical analysis of feature activations
- Process multiple images in batch
- Smart image resizing and padding to maintain aspect ratio
- Comprehensive error handling and reporting

## Requirements

- Python 3.x
- PyTorch
- Ultralytics YOLO
- NumPy
- Matplotlib
- Seaborn
- Pillow (PIL)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install torch ultralytics numpy matplotlib seaborn pillow
```

## Usage

### Basic Usage

```python
from Feature_Extract_Yolo import YOLOFeatureVisualizer

# Initialize the visualizer (defaults to YOLOv8m)
visualizer = YOLOFeatureVisualizer()

# Process a folder of images
results = visualizer.process_al_andalusia("path/to/your/image/folder")
```

### Custom Model

```python
# Use a different YOLO model
visualizer = YOLOFeatureVisualizer(model_name='yolov8s.pt')
```

## Features in Detail

### Feature Map Extraction
- Extracts feature maps from convolutional layers in the first three model blocks
- Processes images while maintaining aspect ratio
- Automatically resizes and pads images to 640x640

### Visualization
- Generates heatmap visualizations for up to 16 channels per layer
- Saves visualizations as PNG files
- Uses viridis colormap for feature map representation

### Statistical Analysis
For each feature map, the tool provides:
- Shape and dimension information
- Basic statistics (mean, std, min, max)
- Per-channel statistics for the first 5 channels
- Activation analysis

## Output

The tool generates:
1. Feature map visualizations saved as PNG files
2. Detailed console output with statistics
3. Processing summary including success/failure counts

## Error Handling

- Tracks failed images with detailed error messages
- Provides summary of processing results
- Gracefully handles various image formats and errors

## Limitations

- Currently supports only convolutional layers in the first three blocks
- Target image size is fixed at 640x640
- Visualizes maximum of 16 channels per layer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your chosen license here]

## Author

[Your name/organization]
