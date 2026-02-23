#!/usr/bin/env python3
"""
Prepare Silero TTS v5 Russian model for iOS mobile deployment.

This script loads the Silero v5_ru model package, extracts the internal
JIT ScriptModule, and saves it for LibTorch Mobile (lite interpreter) on iOS.

The internal JIT model outputs spectral components (magnitude, real, imaginary)
that require ISTFT + PQMF post-processing to become audio. This post-processing
is implemented natively in C++ on iOS using the Accelerate framework, so only
the raw neural network is exported here.

Usage:
    python3 Scripts/trace_model.py

Prerequisites:
    pip install torch numpy scipy
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
    #
    # The internal model outputs spectral data: ((mag, x, y), dur_hat)
    # where mag/x/y are float tensors of shape [1, freq_bins, time_frames].
    # The ISTFT + PQMF conversion to audio is done natively on iOS
    # using the Accelerate framework for maximum compatibility.
    pkg = model.packages[0]
    jit_model = pkg.models[0]

    print(f"Extracted JIT model: {type(jit_model).__name__}")
    print(f"Is ScriptModule: {isinstance(jit_model, torch.jit.ScriptModule)}")
    print()

    # Verify the model produces valid spectral output
    print("Verifying model produces valid output...")
    symbol_to_id = pkg.symbol_to_id
    sos_id = symbol_to_id[pkg.sos_token]  # '|' -> 2
    eos_id = symbol_to_id[pkg.eos_token]  # '~' -> 1

    test_text = "привет"
    tokens = [sos_id]
    for ch in test_text:
        if ch in symbol_to_id:
            tokens.append(symbol_to_id[ch])
    tokens.append(eos_id)

    seq = torch.tensor([tokens], dtype=torch.int64)
    spk = torch.tensor([4], dtype=torch.int64)  # xenia

    with torch.no_grad():
        out, dur_hat = jit_model.forward(seq, spk, 24000)
        mag, x, y = out
        print(f"  mag shape: {mag.shape}, x shape: {x.shape}, y shape: {y.shape}")
        print(f"  dur_hat shape: {dur_hat.shape}")

    # Also verify full pipeline produces audio (using Python post-processing)
    wrapped_jit_v = pkg.wrapped_jit_v
    with torch.no_grad():
        spec = mag * (x + 1j * y)
        audio = wrapped_jit_v.istft(spec).unsqueeze(1)
        audio = wrapped_jit_v.pqmf_2(audio)[:, :1, :].squeeze()
        print(f"  Full pipeline audio: {audio.numel()} samples, range=[{audio.min():.4f}, {audio.max():.4f}]")
        assert audio.numel() > 0, "Pipeline produced empty audio!"
        assert audio.abs().max() > 0.01, "Pipeline produced silent audio!"
    print("  Model verification passed!")
    print()

    # Save with mobile optimization (lite interpreter)
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

        # Verify all speakers work
        print("\nVerifying all speakers...")
        loaded = torch.jit.load(OUTPUT_MODEL)
        speakers = ["aidar", "baya", "kseniya", "eugene", "xenia"]
        for sid, name in enumerate(speakers):
            with torch.no_grad():
                out_v, _ = loaded.forward(seq, torch.tensor([sid], dtype=torch.int64), 24000)
                m, _, _ = out_v
                print(f"  {name} (id={sid}): spectral output shape {m.shape}")
        print("All speakers verified!")

        print()
        print("Next steps:")
        print(f"1. Add {OUTPUT_MODEL} to your Xcode project")
        print("2. Make sure it's included in the app bundle for both targets")
        print("3. Build and run on your iOS device")
        print()
        print("Note: The model outputs spectral data (mag, x, y tensors).")
        print("ISTFT + PQMF post-processing is done natively on iOS")
        print("using the Accelerate framework.")
    else:
        print("Error: Failed to save model!")
        sys.exit(1)


if __name__ == "__main__":
    main()
