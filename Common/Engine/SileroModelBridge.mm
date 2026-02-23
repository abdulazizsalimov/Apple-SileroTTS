#import "SileroModelBridge.h"
#import <LibTorch-Lite/LibTorch-Lite.h>
#include <exception>

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

            // Run inference - the Silero JIT model forward() returns audio
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(sequenceTensor);
            inputs.push_back(speakerTensor);
            inputs.push_back((int64_t)sampleRate);

            auto output = _model->forward(inputs);

            // The output can be a tensor or tuple - handle both cases
            at::Tensor audioTensor;

            if (output.isTensor()) {
                audioTensor = output.toTensor();
            } else if (output.isTuple()) {
                auto tuple = output.toTuple();
                // First element should be audio tensor
                if (tuple->elements()[0].isTensor()) {
                    audioTensor = tuple->elements()[0].toTensor();
                } else if (tuple->elements()[0].isTuple()) {
                    auto inner = tuple->elements()[0].toTuple();
                    audioTensor = inner->elements()[0].toTensor();
                }
            }

            auto audioFlat = audioTensor.flatten().contiguous().to(torch::kFloat32);
            int totalSamples = (int)audioFlat.numel();
            float *audioData = audioFlat.data_ptr<float>();

            NSMutableArray<NSNumber *> *audioArray = [NSMutableArray arrayWithCapacity:totalSamples];
            for (int i = 0; i < totalSamples; i++) {
                [audioArray addObject:@(audioData[i])];
            }

            result = [audioArray copy];
            NSLog(@"[SileroModelBridge] Synthesized %d samples", totalSamples);

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
