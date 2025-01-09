# YOLO Feature Visualizer

A Python tool for extracting and visualizing feature maps from YOLOv8 models. This tool helps researchers and developers understand the internal representations learned by YOLO models by visualizing intermediate feature maps and providing detailed statistical analysis.

<img src="https://github.com/Kygosaur/Feature-Extraction/blob/main/docs/Untitled%20diagram-2025-01-09-083039.png" width="2600"/>

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
  
Example feature map extractions:
<table>
  <tr>
    <td><img src="https://github.com/Kygosaur/Feature-Extraction/blob/main/docs/feature_maps_Test%20(1).jpg_Conv2d_2.png" width="200"/><br>Feature Output</td>
    <td><img src="https://github.com/Kygosaur/Feature-Extraction/blob/main/docs/feature_maps_Test%20(1).jpg_Conv2d_7.png" width="200"/><br>Feature Output</td>
  </tr>
  <tr>
    <td><img src="https://github.com/Kygosaur/Feature-Extraction/blob/main/docs/feature_maps_Test%20(1).jpg_Conv2d_8.png" width="200"/><br>Feature Output</td>
    <td><img src="https://github.com/Kygosaur/Feature-Extraction/blob/main/docs/feature_maps_test%20(1).jpg_Conv2d_9.png" width="200"/><br>Feature Output</td>
  </tr>
</table>

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

## Licensing 

MIT License

Copyright (c) 2025 Kygosaur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

