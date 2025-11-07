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
        from hegarty import Config
        print("  ✓ Config")
    except ImportError as e:
        print(f"  ✗ Config: {e}")
        return False
    
    try:
        from hegarty import GPT4OPerspectiveDetector
        print("  ✓ GPT4OPerspectiveDetector")
    except ImportError as e:
        print(f"  ✗ GPT4OPerspectiveDetector: {e}")
        return False
    
    return True


def test_perspective_detection():
    """Test GPT-4o perspective detection functionality."""
    print("\nTesting GPT-4o perspective detection...")
    
    # Basic import test
    try:
        from hegarty import GPT4OPerspectiveDetector
        from hegarty.gpt_detector import GPTDetectionResult
        print("  ✓ GPT4OPerspectiveDetector can be imported")
        print("  ✓ GPTDetectionResult dataclass available")
        
        # Test that the class has expected methods
        expected_methods = ['analyze', 'detailed_analysis', '_fallback_detection']
        for method in expected_methods:
            if hasattr(GPT4OPerspectiveDetector, method):
                print(f"  ✓ Method '{method}' exists")
            else:
                print(f"  ✗ Method '{method}' missing")
                return False
        
        print("  ℹ Skipping API-dependent tests (requires network access)")
        return True
    except Exception as e:
        print(f"  ✗ Error testing perspective detection: {e}")
        return False


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
    
    # Check if we can import the class
    try:
        print("  ✓ HergartyClient class can be imported")
        
        # Only test initialization if we have a real API key
        if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("dummy"):
            client = HergartyClient()
            print("  ✓ Client initialized successfully with real API key")
        else:
            print("  ℹ Skipping client initialization (requires real OpenAI API key)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
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
