#import "SileroModelBridge.h"
#import <LibTorch/LibTorch.h>
#include <exception>
#include <vector>

// ============================================================================
// SileroModelBridge Implementation
// ============================================================================
// Uses the full LibTorch framework (not Lite) so the exported model can
// include ISTFT + PQMF post-processing and output audio samples directly.

@implementation SileroModelBridge {
    torch::jit::Module _model;
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
        _model = torch::jit::load(path.UTF8String);
        _model.eval();
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
    _model = torch::jit::Module();
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

            // Run inference - the wrapper model returns 1-D audio tensor directly
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(sequenceTensor);
            inputs.push_back(speakerTensor);
            inputs.push_back((int64_t)sampleRate);

            NSLog(@"[SileroModelBridge] Running inference with %d tokens, speaker=%d, sr=%d",
                  seqLen, speakerId, sampleRate);

            auto output = _model.forward(inputs);

            // The wrapper model returns a 1-D float tensor of audio samples
            at::Tensor audioTensor;
            if (output.isTensor()) {
                audioTensor = output.toTensor();
            } else if (output.isTuple()) {
                // Fallback: try first element if tuple
                auto tuple = output.toTuple();
                if (tuple->elements().size() > 0 && tuple->elements()[0].isTensor()) {
                    audioTensor = tuple->elements()[0].toTensor();
                } else {
                    NSLog(@"[SileroModelBridge] Unexpected output format");
                    result = nil;
                    return;
                }
            } else {
                NSLog(@"[SileroModelBridge] Unexpected output type");
                result = nil;
                return;
            }

            audioTensor = audioTensor.contiguous().to(torch::kFloat32);

            int totalSamples = (int)audioTensor.numel();
            if (totalSamples == 0) {
                NSLog(@"[SileroModelBridge] Model returned empty audio");
                result = nil;
                return;
            }

            float *audioData = audioTensor.data_ptr<float>();

            // Convert to NSArray
            NSMutableArray<NSNumber *> *audioArray = [NSMutableArray arrayWithCapacity:totalSamples];
            for (int i = 0; i < totalSamples; i++) {
                [audioArray addObject:@(audioData[i])];
            }

            result = [audioArray copy];
            NSLog(@"[SileroModelBridge] Synthesized %d audio samples (range: [%f, %f])",
                  totalSamples, audioTensor.min().item<float>(), audioTensor.max().item<float>());

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
