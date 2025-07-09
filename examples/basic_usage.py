"""
Basic usage example for RDS Generator
"""

import numpy as np
from PIL import Image
from rds_generator import RDSGenerator, RDSConfig


def main():
    """Basic usage example"""
    
    # Create RDS generator
    generator = RDSGenerator()
    
    # Create configuration
    config = RDSConfig(
        width=512,
        height=512,
        density=50.0,
        disparity_arcsec=30.0,
        distance_cm=57.0,
        ppi=96,
        shape_type='四角形',
        shape_mode='面',
        shape_width=200,
        shape_height=200
    )
    
    print("Generating RDS with configuration:")
    print(f"  Size: {config.width}x{config.height}")
    print(f"  Disparity: {config.disparity_arcsec} arcsec")
    print(f"  Shape: {config.shape_type} ({config.shape_mode})")
    
    # Generate RDS
    result = generator.generate_rds(config)
    
    print(f"Generated RDS with disparity: {result['disparity_pixels']:.2f} pixels")
    
    # Save images
    left_image = Image.fromarray(result['left_image'])
    right_image = Image.fromarray(result['right_image'])
    
    left_image.save('example_left.png')
    right_image.save('example_right.png')
    
    print("Images saved as 'example_left.png' and 'example_right.png'")
    
    # Example with different configurations
    print("\nGenerating batch with different disparities...")
    
    disparities = [-40, -20, 0, 20, 40]
    
    for i, disparity in enumerate(disparities):
        config.disparity_arcsec = disparity
        result = generator.generate_rds(config)
        
        left_image = Image.fromarray(result['left_image'])
        right_image = Image.fromarray(result['right_image'])
        
        sign = "pos" if disparity >= 0 else "neg"
        base_name = f"batch_{sign}{abs(disparity):03d}"
        
        left_image.save(f'{base_name}_L.png')
        right_image.save(f'{base_name}_R.png')
        
        print(f"  Saved {base_name}_L.png and {base_name}_R.png")
    
    print("Batch generation completed!")


if __name__ == "__main__":
    main()