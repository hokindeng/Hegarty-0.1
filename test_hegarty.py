#!/usr/bin/env python3
"""
Quick test script to verify Hegarty installation
"""

import os
import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    from hegarty import HergartyClient
    print("  ✓ HergartyClient")
    
    from hegarty import HergartyAgent
    print("  ✓ HergartyAgent")
    
    from hegarty import Config
    print("  ✓ Config")
    
    from hegarty import GPT4OPerspectiveDetector
    print("  ✓ GPT4OPerspectiveDetector")
    
    return True


def test_perspective_detection():
    """Test GPT-4o perspective detection functionality."""
    print("\nTesting GPT-4o perspective detection...")
    
    # Basic import test
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
    print("  ✓ HergartyClient class can be imported")
    
    # Only test initialization if we have a real API key
    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("dummy"):
        client = HergartyClient()
        print("  ✓ Client initialized successfully with real API key")
    else:
        print("  ℹ Skipping client initialization (requires real OpenAI API key)")
    
    return True


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
        passed = test_func()
        results.append((name, passed))
    
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
