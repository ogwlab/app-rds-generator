# RDS Generator API Documentation

## Overview

The RDS Generator provides a comprehensive API for generating Random Dot Stereograms (RDS) for psychology and vision research.

## Core Classes

### RDSGenerator

The main class for generating RDS images.

```python
from rds_generator import RDSGenerator, RDSConfig

generator = RDSGenerator()
config = RDSConfig(disparity_arcsec=30.0)
result = generator.generate_rds(config)
```

#### Methods

##### `generate_rds(config: RDSConfig) -> Dict[str, np.ndarray]`

Generate RDS from configuration.

**Parameters:**
- `config`: RDSConfig object containing generation parameters

**Returns:**
- Dictionary containing:
  - `left_image`: Left eye image (numpy array)
  - `right_image`: Right eye image (numpy array)
  - `disparity_pixels`: Calculated disparity in pixels
  - `base_image`: Base random dot image
  - `shape_mask`: Boolean mask for shape region

##### `generate_stereo_pair(base_image, disparity_pixels, shape_mask=None) -> Tuple[np.ndarray, np.ndarray]`

Generate stereo pair from base image.

**Parameters:**
- `base_image`: Base random dot image
- `disparity_pixels`: Disparity in pixels
- `shape_mask`: Optional shape mask

**Returns:**
- Tuple of (left_image, right_image)

### RDSConfig

Configuration class for RDS generation parameters.

```python
from rds_generator import RDSConfig

config = RDSConfig(
    width=512,
    height=512,
    density=50.0,
    disparity_arcsec=20.0,
    distance_cm=57.0,
    ppi=96
)
```

#### Parameters

##### Image Parameters
- `width`: Image width in pixels (128-1024)
- `height`: Image height in pixels (128-1024)
- `density`: Dot density percentage (1-100)
- `dot_size`: Size of each dot in pixels (1-10)
- `dot_shape`: Shape of dots ("四角" or "円")
- `bg_color`: Background color (hex string)
- `dot_color`: Dot color (hex string)

##### Stereo Parameters
- `disparity_arcsec`: Disparity in arcseconds (-600 to 600)
- `distance_cm`: Viewing distance in centimeters (30-200)
- `ppi`: Pixels per inch (72-400)

##### Shape Parameters
- `shape_type`: Type of shape ("四角形" or "円")
- `shape_mode`: Display mode ("面" or "枠線")
- `border_width`: Border width for outline mode (1-20)
- `shape_width`: Width of shape in pixels
- `shape_height`: Height of shape in pixels
- `center_x`: Center X coordinate
- `center_y`: Center Y coordinate

#### Methods

##### `to_dict() -> dict`

Convert configuration to dictionary.

##### `from_dict(config_dict: dict) -> RDSConfig`

Create configuration from dictionary.

## Utility Functions

### Mathematical Utilities

#### `arcsec_to_pixels(arcsec: float, distance_cm: float, ppi: int) -> float`

Convert arcseconds to pixels.

#### `apply_phase_shift_2d_robust(image: np.ndarray, shift_x: float, pad_width: int = 32) -> np.ndarray`

Apply robust 2D phase shift using FFT.

### Image Utilities

#### `generate_random_dots(width, height, density, dot_size, dot_shape, bg_color, dot_color) -> np.ndarray`

Generate random dot image.

#### `create_shape_mask(width, height, shape_type, shape_mode, border_width, shape_width, shape_height, center_x, center_y) -> np.ndarray`

Create shape mask for RDS generation.

## Usage Examples

### Basic Usage

```python
from rds_generator import RDSGenerator, RDSConfig

# Create generator
generator = RDSGenerator()

# Configure parameters
config = RDSConfig(
    width=512,
    height=512,
    disparity_arcsec=30.0,
    shape_type='四角形',
    shape_width=200,
    shape_height=200
)

# Generate RDS
result = generator.generate_rds(config)

# Access results
left_image = result['left_image']
right_image = result['right_image']
disparity_pixels = result['disparity_pixels']
```

### Batch Generation

```python
disparities = [-40, -20, 0, 20, 40]

for disparity in disparities:
    config.disparity_arcsec = disparity
    result = generator.generate_rds(config)
    
    # Save or process images
    save_images(result['left_image'], result['right_image'], disparity)
```

### Custom Configuration

```python
# High-resolution RDS with circular shape
config = RDSConfig(
    width=1024,
    height=1024,
    density=75.0,
    disparity_arcsec=50.0,
    distance_cm=50.0,
    ppi=120,
    shape_type='円',
    shape_mode='枠線',
    border_width=5,
    shape_width=400,
    shape_height=400
)

result = generator.generate_rds(config)
```

## Error Handling

The API includes comprehensive validation:

```python
try:
    config = RDSConfig(disparity_arcsec=1000)  # Invalid disparity
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    result = generator.generate_rds(config)
except Exception as e:
    print(f"Generation error: {e}")
```

## Performance Considerations

- Use appropriate image sizes (powers of 2 are more efficient for FFT)
- Batch processing is more efficient than individual image generation
- Cache results when using the same configuration repeatedly
- Consider memory usage for large images or batch processing

## Research Applications

The RDS Generator is designed for:

- Disparity detection threshold measurements
- Developmental vision research
- Stereoscopic depth perception studies
- Binocular vision research
- Psychophysics experiments

For more information, see the README.md file and examples in the examples/ directory.