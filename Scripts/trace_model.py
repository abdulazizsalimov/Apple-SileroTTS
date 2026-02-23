#!/usr/bin/env python3
"""
Prepare Silero TTS v5 Russian model for iOS mobile deployment.

This script loads the Silero v5_ru model package, extracts the
internal JIT ScriptModule, and saves it for LibTorch Mobile on iOS.

Usage:
    python3 Scripts/trace_model.py

Prerequisites:
    pip install torch
    Run ./Scripts/download_model.sh first to download v5_ru.pt
"""

import os
import sys
import torch

MODEL_DIR = "Models"
INPUT_MODEL = os.path.join(MODEL_DIR, "v5_ru.pt")
OUTPUT_MODEL = os.path.join(MODEL_DIR, "silero_v5_ru.ptl")

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("=== Silero TTS Model Preparation for iOS ===")
    print()

    if not os.path.exists(INPUT_MODEL):
        print(f"Error: Model not found at {INPUT_MODEL}")
        print("Run ./Scripts/download_model.sh first to download the model.")
        sys.exit(1)

    print(f"Loading model from {INPUT_MODEL}...")
    package = torch.package.PackageImporter(INPUT_MODEL)
    model = package.load_pickle("tts_models", "model")

    print("Model loaded successfully.")
    print(f"Available speakers: {model.speakers}")
    print(f"Symbols: {model.symbols}")
    print()

    # The Silero v5 model is a wrapper (TTSModelMultiAcc_v3) containing
    # packages, each with JIT ScriptModule models inside.
    # We extract the internal JIT model for LibTorch Mobile.
    pkg = model.packages[0]
    jit_model = pkg.models[0]

    print(f"Extracted JIT model: {type(jit_model).__name__}")
    print(f"Is ScriptModule: {isinstance(jit_model, torch.jit.ScriptModule)}")
    print()

    # Try to save with mobile optimization (lite interpreter)
    print("Saving model for mobile deployment...")
    saved = False

    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized = optimize_for_mobile(jit_model)
        optimized._save_for_lite_interpreter(OUTPUT_MODEL)
        print("Saved optimized model (lite interpreter format)")
        saved = True
    except (ImportError, Exception) as e:
        print(f"Mobile optimization unavailable ({e}), using standard JIT save...")

    if not saved:
        torch.jit.save(jit_model, OUTPUT_MODEL)
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
