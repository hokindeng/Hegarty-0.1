#!/usr/bin/env python3
"""
Interactive demo for Hegarty perspective-taking agent
"""

import os
import sys
from pathlib import Path
from hegarty import HergartyClient, PerspectiveDetector
import base64


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║     HEGARTY - Perspective-Taking Agent Demo          ║
    ║     Enhanced Spatial Reasoning with GPT-4o & Sora    ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def encode_image_file(file_path: str) -> str:
    """Encode an image file to base64."""
    try:
        with open(file_path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def interactive_demo():
    """Run interactive demo."""
    print_banner()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Initialize client
    print("Initializing Hegarty client...")
    try:
        client = HergartyClient()
        detector = PerspectiveDetector()
        print("✓ Client initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Main interaction loop
    print("Enter your questions (type 'help' for commands, 'quit' to exit)")
    print("-" * 60)
    
    current_image = None
    
    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                print("\nGoodbye! 👋")
                break
            
            elif user_input.lower() == 'help':
                print("""
Commands:
  help          - Show this help message
  load <path>   - Load an image file
  clear         - Clear loaded image
  detect <text> - Check if text is a perspective task
  quit/exit     - Exit the demo

Usage:
  - Type any question to process with Hegarty
  - Load an image first for visual questions
  - Questions without images will use text-only mode
                """)
                continue
            
            elif user_input.lower().startswith('load '):
                # Load image
                image_path = user_input[5:].strip()
                if Path(image_path).exists():
                    current_image = encode_image_file(image_path)
                    if current_image:
                        print(f"✓ Image loaded: {image_path}")
                    else:
                        print("❌ Failed to load image")
                else:
                    print(f"❌ File not found: {image_path}")
                continue
            
            elif user_input.lower() == 'clear':
                current_image = None
                print("✓ Image cleared")
                continue
            
            elif user_input.lower().startswith('detect '):
                # Detect if text is perspective task
                text = user_input[7:].strip()
                is_perspective, confidence = detector.analyze(text)
                
                if is_perspective:
                    print(f"✓ This IS a perspective-taking task (confidence: {confidence:.2f})")
                else:
                    print(f"✗ This is NOT a perspective-taking task (confidence: {confidence:.2f})")
                continue
            
            # Process question with Hegarty
            print("\nProcessing...")
            
            # Check if this is a perspective task
            is_perspective, confidence = detector.analyze(user_input)
            
            if is_perspective:
                print(f"🔄 Detected perspective-taking task (confidence: {confidence:.2f})")
                model = "hegarty-1.0"
            else:
                print(f"📝 Standard question detected")
                model = "gpt-4o"
            
            # Build message
            if current_image:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": current_image}}
                    ]
                }]
                print("   Using loaded image")
            else:
                messages = [{
                    "role": "user",
                    "content": user_input
                }]
            
            # Get response
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                
                print("\nAnswer:")
                print("-" * 60)
                print(response.choices[0].message.content)
                print("-" * 60)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Handle command line arguments
        if sys.argv[1] == '--help':
            print("""
Hegarty Demo - Interactive perspective-taking agent

Usage:
  hegarty-demo              Run interactive demo
  hegarty-demo --help       Show this help message
  hegarty-demo --version    Show version

Environment Variables:
  OPENAI_API_KEY           Required: Your OpenAI API key
  SORA_API_KEY            Optional: Sora API key (uses simulation if not set)
            """)
        elif sys.argv[1] == '--version':
            print("Hegarty version 0.1.0")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Run interactive demo
        interactive_demo()


if __name__ == "__main__":
    main()
