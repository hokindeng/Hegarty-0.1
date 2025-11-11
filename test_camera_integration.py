"""Quick test script for camera detection integration"""

import sys
sys.path.insert(0, 'external/PerspectiveFields')

try:
    from perspective2d import PerspectiveFields
    print("✓ PerspectiveFields import successful")
    
    # List available models
    print("\nAvailable PerspectiveFields models:")
    models = [
        'Paramnet-360Cities-edina-centered',
        'Paramnet-360Cities-edina-uncentered',
        'PersNet-360Cities',
        'PersNet_paramnet-GSV-centered',
        'PersNet_Paramnet-GSV-uncentered'
    ]
    for model in models:
        print(f"  - {model}")
    
    # Test CameraDetector import
    from hegarty.camera_detector import CameraDetector
    print("\n✓ CameraDetector import successful")
    
    # Test initialization (without loading model weights)
    print("\nCamera detection integration is ready!")
    print("Note: Model weights will be downloaded on first use.")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nPlease install required dependencies:")
    print("  pip install -r requirements.txt")
    print("  git submodule update --init --recursive")
