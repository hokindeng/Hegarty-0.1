#!/usr/bin/env python3
"""
Quick test script to verify Hegarty installation
"""

import os
import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from hegarty import HergartyClient
        print("  ✓ HergartyClient")
    except ImportError as e:
        print(f"  ✗ HergartyClient: {e}")
        return False
    
    try:
        from hegarty import HergartyAgent
        print("  ✓ HergartyAgent")
    except ImportError as e:
        print(f"  ✗ HergartyAgent: {e}")
        return False
    
    try:
        from hegarty import PerspectiveDetector
        print("  ✓ PerspectiveDetector")
    except ImportError as e:
        print(f"  ✗ PerspectiveDetector: {e}")
        return False
    
    try:
        from hegarty import Config
        print("  ✓ Config")
    except ImportError as e:
        print(f"  ✗ Config: {e}")
        return False
    
    return True


def test_perspective_detection():
    """Test perspective detection functionality."""
    print("\nTesting perspective detection...")
    
    from hegarty import PerspectiveDetector
    
    detector = PerspectiveDetector()
    
    test_cases = [
        ("Rotate this 90 degrees", True),
        ("What is the weather today?", False),
        ("Flip this image horizontally", True),
        ("Calculate 2+2", False)
    ]
    
    all_passed = True
    for text, expected in test_cases:
        is_perspective, confidence = detector.analyze(text)
        passed = is_perspective == expected
        
        if passed:
            print(f"  ✓ '{text[:30]}...' -> {is_perspective} (expected {expected})")
        else:
            print(f"  ✗ '{text[:30]}...' -> {is_perspective} (expected {expected})")
            all_passed = False
    
    return all_passed


def test_config():
    """Test configuration management."""
    print("\nTesting configuration...")
    
    from hegarty import Config
    
    # Test default config
    config = Config()
    print(f"  ✓ Default config created")
    print(f"    - Model: {config.gpt_model}")
    print(f"    - Temperature: {config.temperature}")
    
    # Test custom config
    custom_config = Config(temperature=0.5, max_workers=8)
    assert custom_config.temperature == 0.5
    assert custom_config.max_workers == 8
    print(f"  ✓ Custom config works")
    
    return True


def test_client_initialization():
    """Test client initialization (without API calls)."""
    print("\nTesting client initialization...")
    
    from hegarty import HergartyClient
    
    # Set dummy API key if not present
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
        print("  ℹ Using dummy API key for testing")
    
    try:
        client = HergartyClient()
        print("  ✓ Client initialized successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize client: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Hegarty Package Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Perspective Detection", test_perspective_detection),
        ("Configuration", test_config),
        ("Client Initialization", test_client_initialization),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("-" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
