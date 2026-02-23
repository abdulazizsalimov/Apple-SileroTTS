#!/usr/bin/env python3
"""
Trace Silero TTS v5 Russian model for iOS mobile deployment.

This script loads the PyTorch model and traces it for use with
LibTorch Mobile on iOS devices.

Usage:
    python3 Scripts/trace_model.py

Prerequisites:
    pip install torch torchaudio
"""

import os
import sys
import torch

MODEL_DIR = "Models"
INPUT_MODEL = os.path.join(MODEL_DIR, "v5_ru.pt")
OUTPUT_MODEL = os.path.join(MODEL_DIR, "silero_v5_ru.ptl")


def main():
    print("=== Silero TTS Model Tracer for iOS ===")
    print()

    # Check input model exists
    if not os.path.exists(INPUT_MODEL):
        print(f"Error: Model not found at {INPUT_MODEL}")
        print("Run ./Scripts/download_model.sh first to download the model.")
        sys.exit(1)

    print(f"Loading model from {INPUT_MODEL}...")
    model = torch.package.PackageImporter(INPUT_MODEL).load_pickle("tts_models", "model")
    model.eval()

    print("Model loaded successfully.")
    print(f"Available speakers: {model.speakers}")

    # Trace the model for mobile
    print()
    print("Tracing model for mobile deployment...")

    # Use the model's built-in method to save for mobile if available
    # Silero v5 models support save_for_mobile
    if hasattr(model, 'save'):
        # Create a traced version
        example_text = "Привет мир"
        
        # Try to use the model's own tracing
        try:
            traced = torch.jit.trace(model, example_args=None)
        except Exception:
            pass

    # Use torch.jit.script or the model's own export
    print("Saving optimized model for mobile...")
    
    # For Silero models, we can use torch.jit.save with optimization
    scripted = torch.jit.script(model)
    optimized = torch.utils.mobile_optimizer.optimize_for_mobile(scripted)
    optimized._save_for_lite_interpreter(OUTPUT_MODEL)

    if os.path.exists(OUTPUT_MODEL):
        size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
        print(f"Model saved to {OUTPUT_MODEL} ({size_mb:.1f} MB)")
        print()
        print("Next steps:")
        print(f"1. Add {OUTPUT_MODEL} to your Xcode project")
        print("2. Make sure it's included in the app bundle for both targets")
        print("3. Build and run on your iOS device")
    else:
        print("Error: Failed to save traced model!")
        sys.exit(1)


if __name__ == "__main__":
    main()
