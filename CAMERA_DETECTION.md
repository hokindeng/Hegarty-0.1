# Camera Detection with PerspectiveFields

Hegarty now integrates [PerspectiveFields](https://github.com/jinlinyi/PerspectiveFields) for automatic camera parameter detection. This improves the quality of generated videos by using accurate camera angles based on the input image.

## Features

- **Automatic Camera Parameter Detection**: Detects roll, pitch, and vertical field of view from input images
- **Smart Prompt Generation**: Uses detected camera parameters to create precise Sora prompts
- **Multiple Model Support**: Choose from various PerspectiveFields models for different scenarios

## Setup

1. **Initialize the submodule**:
```bash
git submodule update --init --recursive
```

2. **Install additional dependencies**:
```bash
pip install -r requirements.txt
```

## Configuration

Camera detection is enabled by default in the config:

```python
from hegarty import Config

config = Config(
    use_camera_detection=True,  # Enable camera detection
    camera_detector_model="Paramnet-360Cities-edina-centered",  # Model choice
    camera_detection_confidence_threshold=0.5  # Minimum confidence
)
```

### Available Models

- `Paramnet-360Cities-edina-centered`: Best for uncropped indoor/outdoor/natural images
- `Paramnet-360Cities-edina-uncentered`: Best for cropped images with off-center principal points
- `PersNet-360Cities`: Basic perspective field detection
- `PersNet_paramnet-GSV-centered`: Optimized for street view images
- `PersNet_Paramnet-GSV-uncentered`: Street view with cropping support

## How It Works

1. **Detection**: When processing an image, PerspectiveFields analyzes it to detect:
   - Camera roll (rotation around Z-axis)
   - Camera pitch/elevation (rotation around X-axis)
   - Vertical field of view (vFOV)
   - Perspective fields (up vectors and latitude)

2. **Prompt Enhancement**: Detected parameters are used to generate precise Sora prompts:
   ```
   First frame: Your camera is tilted at 25° elevation, viewing from 0° azimuth.
   Final frame: Your camera remains at 25° elevation, but rotates horizontally to 180° azimuth.
   Create a smooth video showing the camera's horizontal rotation around the teddy bear,
   and try to maintain the tilted viewing angle throughout.
   ```

3. **Fallback**: If detection confidence is low (< 0.5), default parameters are used.

## Example Usage

```python
from hegarty import HergartyAgent, Config
from hegarty.mllm import QwenMLLM

# Configure with camera detection
config = Config(
    use_camera_detection=True,
    camera_detector_model="Paramnet-360Cities-edina-centered"
)

# Initialize agent
agent = HergartyAgent(config=config)
agent.mllm = QwenMLLM()

# Process image - camera detection happens automatically
result = agent.process(
    image="path/to/image.jpg",
    question="What would this look like from the opposite side?"
)
```

## Direct Camera Detection

You can also use the camera detector directly:

```python
from hegarty.camera_detector import CameraDetector
from PIL import Image

# Initialize detector
detector = CameraDetector(model_version='Paramnet-360Cities-edina-centered')

# Detect parameters
image = Image.open("path/to/image.jpg")
params = detector.detect_from_image(image)

print(f"Roll: {params['roll']:.1f}°")
print(f"Pitch: {params['pitch']:.1f}°")
print(f"vFOV: {params['vfov']:.1f}°")
print(f"Confidence: {params['confidence']:.2f}")
```

## Benefits

1. **More Accurate Videos**: Camera movements match the original image perspective
2. **Better Stabilization**: Maintains consistent elevation throughout rotation
3. **Automatic Adjustment**: Adapts to different image types and perspectives
4. **Quality Control**: Confidence scores ensure reliable results

## Troubleshooting

If you encounter import errors:
```bash
# Ensure submodule is initialized
git submodule update --init --recursive

# Install missing dependencies
pip install yacs timm albumentations scipy scikit-learn omegaconf pyequilib==0.3.0
```

## Citation

If you use the camera detection feature, please also cite PerspectiveFields:

```bibtex
@inproceedings{jin2023perspective,
    title={Perspective Fields for Single Image Camera Calibration},
    author={Jin, Linyi and Zhang, Jianming and Hold-Geoffroy, Yannick and Wang, Oliver and Matzen, Kevin and Sticha, Matthew and Fouhey, David F.},
    booktitle={CVPR},
    year={2023}
}
```
