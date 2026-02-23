#!/usr/bin/env python3
"""
Prepare Silero TTS v5 Russian model for iOS mobile deployment.

This script loads the Silero model via torch.hub and saves it
in a format suitable for LibTorch Mobile on iOS.

Usage:
    python3 Scripts/trace_model.py

Prerequisites:
    pip install torch numpy omegaconf
"""

import os
import sys
import torch

MODEL_DIR = "Models"
OUTPUT_MODEL = os.path.join(MODEL_DIR, "silero_v5_ru.ptl")

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("=== Silero TTS Model Preparation for iOS ===")
    print()

    print("Loading Silero v5 Russian model via torch.hub...")
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v5_ru'
    )
    model.eval()

    print("Model loaded successfully.")
    print(f"Available speakers: {model.speakers}")
    print()

    # Test synthesis to verify model works
    print("Testing model with sample text...")
    test_audio = model.apply_tts(
        text="Привет мир",
        speaker="aidar",
        sample_rate=24000
    )
    print(f"Test synthesis OK: {test_audio.shape[0]} samples generated")
    print()

    # Save model for mobile deployment
    print("Saving model for mobile deployment...")

    # The Silero model loaded via torch.hub is already a ScriptModule
    # We can optimize it for mobile and save
    try:
        optimized = torch.utils.mobile_optimizer.optimize_for_mobile(model)
        optimized._save_for_lite_interpreter(OUTPUT_MODEL)
        print("Saved optimized model (lite interpreter format)")
    except Exception as e:
        print(f"Mobile optimization failed ({e}), saving as standard JIT...")
        torch.jit.save(model, OUTPUT_MODEL)
        print("Saved as standard JIT model")

    if os.path.exists(OUTPUT_MODEL):
        size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
        print(f"\nModel saved to {OUTPUT_MODEL} ({size_mb:.1f} MB)")
        print()
        print("Next steps:")
        print(f"1. Add {OUTPUT_MODEL} to your Xcode project")
        print("2. Make sure it's included in the app bundle for both targets")
        print("3. Build and run on your iOS device")
    else:
        print("Error: Failed to save model!")
        sys.exit(1)


if __name__ == "__main__":
    main()
