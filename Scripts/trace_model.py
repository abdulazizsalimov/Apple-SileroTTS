#!/usr/bin/env python3
"""
Prepare Silero TTS v5 Russian model for iOS mobile deployment.

This script loads the Silero v5_ru model package, extracts the internal
JIT ScriptModule, wraps it with ISTFT + PQMF post-processing so the
exported model outputs audio samples directly, and saves it as a standard
TorchScript model for the full LibTorch framework on iOS.

Usage:
    python3 Scripts/trace_model.py

Prerequisites:
    pip install torch numpy scipy
    Run ./Scripts/download_model.sh first to download v5_ru.pt
"""

import os
import sys
import torch
import torch.nn as nn

MODEL_DIR = "Models"
INPUT_MODEL = os.path.join(MODEL_DIR, "v5_ru.pt")
OUTPUT_MODEL = os.path.join(MODEL_DIR, "silero_v5_ru.ptl")

os.makedirs(MODEL_DIR, exist_ok=True)


class SileroMobileWrapper(nn.Module):
    """Wraps the internal Silero JIT model + ISTFT + PQMF into a single
    module whose forward() returns a 1-D float tensor of audio samples.

    The raw JIT model outputs (magnitude, real, imaginary) spectral tensors.
    This wrapper applies ISTFT reconstruction and PQMF analysis so callers
    get playback-ready audio directly.
    """

    def __init__(self, jit_model, istft, pqmf_2):
        super().__init__()
        self.jit_model = jit_model
        self.istft = istft
        self.pqmf_2 = pqmf_2

    def forward(
        self,
        sequence: torch.Tensor,
        speaker_ids: torch.Tensor,
        sr: int = 24000,
    ) -> torch.Tensor:
        out, dur_hat = self.jit_model(sequence, speaker_ids, sr)
        mag, x, y = out
        mag = mag.to("cpu")
        x = x.to("cpu")
        y = y.to("cpu")
        audio = self.istft(mag * (x + 1j * y)).unsqueeze(1)
        if sr == 24000:
            audio = self.pqmf_2(audio)[:, :1, :]
        audio = audio.squeeze(1).squeeze(0)
        return audio


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

    # Extract the internal JIT model and post-processing modules
    pkg = model.packages[0]
    jit_model = pkg.models[0]
    wrapped_jit_v = pkg.wrapped_jit_v

    print(f"Extracted JIT model: {type(jit_model).__name__}")
    print(f"Post-processing: ISTFT + PQMF from WrappedJitV")
    print()

    # Create wrapper that bundles model + ISTFT + PQMF
    wrapper = SileroMobileWrapper(jit_model, wrapped_jit_v.istft, wrapped_jit_v.pqmf_2)
    wrapper.eval()

    # Verify the wrapper produces correct audio
    print("Verifying wrapper produces audio...")
    symbol_to_id = pkg.symbol_to_id
    sos_id = symbol_to_id[pkg.sos_token]
    eos_id = symbol_to_id[pkg.eos_token]

    test_text = "привет"
    tokens = [sos_id]
    for ch in test_text:
        if ch in symbol_to_id:
            tokens.append(symbol_to_id[ch])
    tokens.append(eos_id)

    seq = torch.tensor([tokens], dtype=torch.int64)
    spk = torch.tensor([4], dtype=torch.int64)

    with torch.no_grad():
        audio = wrapper(seq, spk, 24000)
    print(f"  Test audio: {audio.numel()} samples, range=[{audio.min():.4f}, {audio.max():.4f}]")
    assert audio.numel() > 0, "Wrapper produced empty audio!"
    assert audio.abs().max() > 0.01, "Wrapper produced silent audio!"
    print("  Verification passed!")
    print()

    # Script the wrapper for serialization
    print("Scripting wrapper...")
    scripted = torch.jit.script(wrapper)

    with torch.no_grad():
        audio2 = scripted(seq, spk, 24000)
    print(f"  Scripted audio: {audio2.numel()} samples, range=[{audio2.min():.4f}, {audio2.max():.4f}]")
    assert audio2.numel() > 0
    print("  Scripted model OK!")
    print()

    # Save as standard TorchScript (for full LibTorch, not lite interpreter)
    print("Saving model...")
    torch.jit.save(scripted, OUTPUT_MODEL)

    if os.path.exists(OUTPUT_MODEL):
        size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
        print(f"Model saved to {OUTPUT_MODEL} ({size_mb:.1f} MB)")

        # Verify all speakers
        print("\nVerifying all speakers...")
        loaded = torch.jit.load(OUTPUT_MODEL)
        speakers = ["aidar", "baya", "kseniya", "eugene", "xenia"]
        for sid, name in enumerate(speakers):
            with torch.no_grad():
                a = loaded(seq, torch.tensor([sid], dtype=torch.int64), 24000)
            print(f"  {name} (id={sid}): {a.numel()} samples, range=[{a.min():.4f}, {a.max():.4f}]")
        print("All speakers verified!")
    else:
        print("Error: Failed to save model!")
        sys.exit(1)


if __name__ == "__main__":
    main()
