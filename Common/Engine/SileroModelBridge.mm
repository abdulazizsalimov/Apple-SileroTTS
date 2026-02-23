#import "SileroModelBridge.h"
#import <LibTorch-Lite/LibTorch-Lite.h>
#import <Accelerate/Accelerate.h>
#include <exception>
#include <vector>
#include <cmath>

// ============================================================================
// ISTFT + PQMF Post-Processing Constants
// ============================================================================
// The Silero JIT model outputs spectral data (mag, x, y tensors), not audio.
// We apply ISTFT (inverse Short-Time Fourier Transform) and PQMF (Pseudo
// Quadrature Mirror Filter) analysis natively to convert to audio samples.
// This avoids requiring FFT ops in LibTorch-Lite which may not be available.

// ISTFT parameters (from the Silero ISTFT module)
static const int kISTFT_NFFT = 2400;
static const int kISTFT_HopLength = 600;
static const int kISTFT_WinLength = 2400;
static const int kISTFT_FreqBins = 1201;  // n_fft/2 + 1
static const int kISTFT_Pad = 900;        // (win_length - hop_length) / 2

// PQMF parameters for 24kHz (from the Silero PQMF module)
static const int kPQMF_N = 2;
static const int kPQMF_FilterLen = 63;
static const int kPQMF_Padding = 31;

// PQMF Analysis Filter H[0] - pre-computed from the model
// N=2, taps=62, cutoff=0.25, beta=10.0
static const float kPQMF_H0[63] = {
    1.9735623482e-06f, 2.3249980586e-05f, 3.8659974962e-05f, -3.8572824599e-20f,
    5.7133762311e-05f, 3.2661485602e-04f, 3.6693888251e-04f, -2.4075585452e-19f,
    3.4139165655e-04f, 1.6692556674e-03f, 1.6496476019e-03f, -8.0964029936e-19f,
    1.2557493756e-03f, 5.6645874865e-03f, 5.2156210877e-03f, -1.9238254426e-18f,
    3.5292678513e-03f, 1.5162172727e-02f, 1.3376200572e-02f, -3.5663978964e-18f,
    8.4574390203e-03f, 3.5448815674e-02f, 3.0726540834e-02f, -5.4107587859e-18f,
    1.9265413284e-02f, 8.1937320530e-02f, 7.3466211557e-02f, -6.8902697460e-18f,
    5.4921995848e-02f, 2.8832343221e-01f, 4.1384068131e-01f, 1.9134026766e-01f,
   -1.7141842842e-01f, -2.8832343221e-01f, -1.3259342313e-01f, -6.8902697460e-18f,
   -3.0430700630e-02f, -8.1937320530e-02f, -4.6510819346e-02f, -5.4107587859e-18f,
   -1.2727349997e-02f, -3.5448815674e-02f, -2.0418064669e-02f, -3.5663978964e-18f,
   -5.5406037718e-03f, -1.5162172727e-02f, -8.5204066709e-03f, -1.9238254426e-18f,
   -2.1603810601e-03f, -5.6645874865e-03f, -3.0316472985e-03f, -8.0964029936e-19f,
   -6.8330636714e-04f, -1.6692556674e-03f, -8.2419236423e-04f, -2.4075585452e-19f,
   -1.5199105837e-04f, -3.2661485602e-04f, -1.3793310791e-04f, -3.8572824599e-20f,
   -1.6013485947e-05f, -2.3249980586e-05f, -4.7646008170e-06f
};

// ============================================================================
// Native ISTFT + PQMF Implementation
// ============================================================================

/// Compute Hann window of given length: w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
static std::vector<float> computeHannWindow(int length) {
    std::vector<float> window(length);
    for (int i = 0; i < length; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)i / (float)(length - 1)));
    }
    return window;
}

/// Perform inverse real FFT: 1201 complex bins -> 2400 real samples
/// Uses Accelerate's vDSP_DFT for non-power-of-2 sizes.
static void performIRFFT(const float *realIn, const float *imagIn,
                         float *realOut, int freqBins, int nfft,
                         vDSP_DFT_Setup dftSetup) {
    // Extend one-sided spectrum to full complex spectrum via conjugate symmetry
    // For k = 0..freqBins-1: use input directly
    // For k = freqBins..nfft-1: conjugate mirror
    std::vector<float> fullReal(nfft);
    std::vector<float> fullImag(nfft);

    for (int k = 0; k < freqBins; k++) {
        fullReal[k] = realIn[k];
        fullImag[k] = imagIn[k];
    }
    for (int k = freqBins; k < nfft; k++) {
        int mirror = nfft - k;
        fullReal[k] = realIn[mirror];
        fullImag[k] = -imagIn[mirror];
    }

    // Inverse DFT
    std::vector<float> outReal(nfft);
    std::vector<float> outImag(nfft);
    vDSP_DFT_Execute(dftSetup, fullReal.data(), fullImag.data(),
                     outReal.data(), outImag.data());

    // Scale by 1/N (vDSP inverse DFT doesn't normalize)
    float scale = 1.0f / (float)nfft;
    vDSP_vsmul(outReal.data(), 1, &scale, realOut, 1, nfft);
}

/// Apply ISTFT to spectral data (mag * x, mag * y) -> audio samples
/// Input: mag, x, y tensors each of shape [1, freqBins, timeFrames]
/// Output: audio samples vector
static std::vector<float> applyISTFT(const float *mag, const float *x, const float *y,
                                     int freqBins, int timeFrames) {
    int nfft = kISTFT_NFFT;
    int hopLength = kISTFT_HopLength;
    int winLength = kISTFT_WinLength;
    int pad = kISTFT_Pad;

    // Compute Hann window
    auto window = computeHannWindow(winLength);

    // Output size
    int outputSize = (timeFrames - 1) * hopLength + winLength;
    std::vector<float> output(outputSize, 0.0f);
    std::vector<float> windowEnvelope(outputSize, 0.0f);

    // Create DFT setup for inverse FFT
    vDSP_DFT_Setup dftSetup = vDSP_DFT_zop_CreateSetup(
        NULL, (vDSP_Length)nfft, vDSP_DFT_INVERSE
    );
    if (!dftSetup) {
        NSLog(@"[SileroModelBridge] Failed to create DFT setup");
        return {};
    }

    // Temporary buffers for each frame
    std::vector<float> frameReal(freqBins);
    std::vector<float> frameImag(freqBins);
    std::vector<float> frameSignal(nfft);

    for (int t = 0; t < timeFrames; t++) {
        // Compute complex spectrogram for this frame:
        // real_part = mag[f,t] * x[f,t], imag_part = mag[f,t] * y[f,t]
        // Tensor layout is [1, freqBins, timeFrames] in row-major order
        for (int f = 0; f < freqBins; f++) {
            int idx = f * timeFrames + t;  // [0, f, t] in row-major
            float m = mag[idx];
            frameReal[f] = m * x[idx];
            frameImag[f] = m * y[idx];
        }

        // Inverse FFT: freqBins complex -> nfft real
        performIRFFT(frameReal.data(), frameImag.data(),
                     frameSignal.data(), freqBins, nfft, dftSetup);

        // Apply Hann window and overlap-add
        int start = t * hopLength;
        for (int i = 0; i < winLength; i++) {
            float windowed = frameSignal[i] * window[i];
            output[start + i] += windowed;
            windowEnvelope[start + i] += window[i] * window[i];
        }
    }

    vDSP_DFT_DestroySetup(dftSetup);

    // Normalize by window envelope
    for (int i = 0; i < outputSize; i++) {
        if (windowEnvelope[i] > 1e-11f) {
            output[i] /= windowEnvelope[i];
        }
    }

    // Trim padding
    int trimmedSize = outputSize - 2 * pad;
    if (trimmedSize <= 0) {
        return {};
    }
    std::vector<float> trimmed(output.begin() + pad, output.begin() + pad + trimmedSize);
    return trimmed;
}

/// Apply PQMF analysis (strided convolution with H[0] filter)
/// Input: audio samples
/// Output: downsampled audio (samples/stride)
static std::vector<float> applyPQMF(const std::vector<float> &input) {
    int stride = kPQMF_N;
    int padding = kPQMF_Padding;
    int filterLen = kPQMF_FilterLen;
    int inputLen = (int)input.size();

    // Output length: floor((inputLen + 2*padding - filterLen) / stride) + 1
    int outputLen = (inputLen + 2 * padding - filterLen) / stride + 1;
    if (outputLen <= 0) return {};

    std::vector<float> output(outputLen, 0.0f);

    for (int i = 0; i < outputLen; i++) {
        float sum = 0.0f;
        int inputStart = i * stride - padding;
        for (int j = 0; j < filterLen; j++) {
            int inputIdx = inputStart + j;
            if (inputIdx >= 0 && inputIdx < inputLen) {
                sum += input[inputIdx] * kPQMF_H0[j];
            }
        }
        output[i] = sum;
    }

    // Clamp: if max(abs(output)) > 1.0, apply tanh
    float maxAbs = 0.0f;
    for (int i = 0; i < outputLen; i++) {
        float a = fabsf(output[i]);
        if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs > 1.0f) {
        for (int i = 0; i < outputLen; i++) {
            output[i] = tanhf(output[i]);
        }
    }

    return output;
}

// ============================================================================
// SileroModelBridge Implementation
// ============================================================================

@implementation SileroModelBridge {
    std::unique_ptr<torch::jit::mobile::Module> _model;
    BOOL _loaded;
    dispatch_queue_t _inferenceQueue;
}

+ (SileroModelBridge *)shared {
    static SileroModelBridge *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[SileroModelBridge alloc] init];
    });
    return instance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _loaded = NO;
        _inferenceQueue = dispatch_queue_create("com.silero.tts.inference", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (BOOL)isModelLoaded {
    return _loaded;
}

- (BOOL)loadModelAtPath:(NSString *)path {
    try {
        _model = std::make_unique<torch::jit::mobile::Module>(
            torch::jit::_load_for_mobile(path.UTF8String)
        );
        _model->eval();
        _loaded = YES;
        NSLog(@"[SileroModelBridge] Model loaded from: %@", path);
        return YES;
    } catch (const c10::Error& e) {
        NSLog(@"[SileroModelBridge] LibTorch error loading model: %s", e.what());
        _loaded = NO;
        return NO;
    } catch (const std::exception& e) {
        NSLog(@"[SileroModelBridge] C++ exception loading model: %s", e.what());
        _loaded = NO;
        return NO;
    } catch (...) {
        NSLog(@"[SileroModelBridge] Unknown exception loading model");
        _loaded = NO;
        return NO;
    }
}

- (void)unloadModel {
    _model.reset();
    _loaded = NO;
    NSLog(@"[SileroModelBridge] Model unloaded");
}

- (nullable NSArray<NSNumber *> *)synthesizeWithTokens:(NSArray<NSNumber *> *)tokens
                                             speakerId:(int)speakerId
                                            sampleRate:(int)sampleRate {
    if (!_loaded) {
        NSLog(@"[SileroModelBridge] Model not loaded");
        return nil;
    }

    __block NSArray<NSNumber *> *result = nil;

    dispatch_sync(_inferenceQueue, ^{
        try {
            if (!_model) {
                NSLog(@"[SileroModelBridge] Model pointer is null");
                result = nil;
                return;
            }

            // Create sequence tensor [1, seq_len]
            int seqLen = (int)tokens.count;
            auto sequenceTensor = torch::zeros({1, seqLen}, torch::kInt64);
            auto seqAccessor = sequenceTensor.accessor<int64_t, 2>();
            for (int i = 0; i < seqLen; i++) {
                seqAccessor[0][i] = tokens[i].intValue;
            }

            // Create speaker_ids tensor [1]
            auto speakerTensor = torch::zeros({1}, torch::kInt64);
            speakerTensor[0] = speakerId;

            // Run inference - the raw JIT model returns spectral data:
            // output = ((mag, x, y), dur_hat)
            // where mag/x/y are float tensors of shape [1, freq_bins, time_frames]
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(sequenceTensor);
            inputs.push_back(speakerTensor);
            inputs.push_back((int64_t)sampleRate);

            NSLog(@"[SileroModelBridge] Running inference with %d tokens, speaker=%d, sr=%d",
                  seqLen, speakerId, sampleRate);

            auto output = _model->forward(inputs);

            // Extract (mag, x, y) from nested tuple output:
            // output is Tuple[Tuple[mag, x, y], dur_hat]
            if (!output.isTuple()) {
                NSLog(@"[SileroModelBridge] Expected tuple output, got something else");
                result = nil;
                return;
            }

            auto outerTuple = output.toTuple();
            if (outerTuple->elements().size() < 2) {
                NSLog(@"[SileroModelBridge] Outer tuple has %zu elements, expected 2",
                      outerTuple->elements().size());
                result = nil;
                return;
            }

            // First element is the inner tuple (mag, x, y)
            auto spectralData = outerTuple->elements()[0];
            if (!spectralData.isTuple()) {
                NSLog(@"[SileroModelBridge] Expected inner tuple for spectral data");
                result = nil;
                return;
            }

            auto innerTuple = spectralData.toTuple();
            if (innerTuple->elements().size() < 3) {
                NSLog(@"[SileroModelBridge] Inner tuple has %zu elements, expected 3",
                      innerTuple->elements().size());
                result = nil;
                return;
            }

            // Extract mag, x, y tensors
            auto magTensor = innerTuple->elements()[0].toTensor().contiguous().to(torch::kFloat32);
            auto xTensor = innerTuple->elements()[1].toTensor().contiguous().to(torch::kFloat32);
            auto yTensor = innerTuple->elements()[2].toTensor().contiguous().to(torch::kFloat32);

            // Verify shapes: [1, freqBins, timeFrames]
            int freqBins = (int)magTensor.size(1);
            int timeFrames = (int)magTensor.size(2);

            NSLog(@"[SileroModelBridge] Spectral output: freqBins=%d, timeFrames=%d",
                  freqBins, timeFrames);

            // Get raw data pointers
            float *magData = magTensor.data_ptr<float>();
            float *xData = xTensor.data_ptr<float>();
            float *yData = yTensor.data_ptr<float>();

            // Apply ISTFT: spectral data -> audio waveform
            auto istftAudio = applyISTFT(magData, xData, yData, freqBins, timeFrames);
            if (istftAudio.empty()) {
                NSLog(@"[SileroModelBridge] ISTFT produced empty output");
                result = nil;
                return;
            }

            NSLog(@"[SileroModelBridge] ISTFT produced %d samples", (int)istftAudio.size());

            // Apply PQMF analysis for 24kHz sample rate
            std::vector<float> finalAudio;
            if (sampleRate == 24000) {
                finalAudio = applyPQMF(istftAudio);
                NSLog(@"[SileroModelBridge] PQMF produced %d samples", (int)finalAudio.size());
            } else {
                finalAudio = std::move(istftAudio);
            }

            if (finalAudio.empty()) {
                NSLog(@"[SileroModelBridge] Final audio is empty");
                result = nil;
                return;
            }

            // Convert to NSArray
            int totalSamples = (int)finalAudio.size();
            NSMutableArray<NSNumber *> *audioArray = [NSMutableArray arrayWithCapacity:totalSamples];
            for (int i = 0; i < totalSamples; i++) {
                [audioArray addObject:@(finalAudio[i])];
            }

            result = [audioArray copy];
            NSLog(@"[SileroModelBridge] Synthesized %d audio samples", totalSamples);

        } catch (const c10::Error& e) {
            NSLog(@"[SileroModelBridge] LibTorch inference error: %s", e.what());
            result = nil;
        } catch (const std::exception& e) {
            NSLog(@"[SileroModelBridge] C++ inference error: %s", e.what());
            result = nil;
        } catch (...) {
            NSLog(@"[SileroModelBridge] Unknown inference error");
            result = nil;
        }
    });

    return result;
}

@end
