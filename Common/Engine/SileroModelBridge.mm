#import "SileroModelBridge.h"
#import <Libtorch-Lite/Libtorch-Lite.h>

@implementation SileroModelBridge {
    torch::jit::mobile::Module _model;
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
    @try {
        _model = torch::jit::_load_for_mobile(path.UTF8String);
        _model.eval();
        _loaded = YES;
        NSLog(@"[SileroModelBridge] Model loaded from: %@", path);
        return YES;
    } @catch (NSException *exception) {
        NSLog(@"[SileroModelBridge] Failed to load model: %@", exception.reason);
        _loaded = NO;
        return NO;
    }
}

- (void)unloadModel {
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
        @try {
            // Create sequence tensor [1, seq_len]
            int seqLen = (int)tokens.count;
            auto sequenceTensor = torch::zeros({1, seqLen}, torch::kInt64);
            auto sequenceAccessor = sequenceTensor.accessor<int64_t, 2>();
            for (int i = 0; i < seqLen; i++) {
                sequenceAccessor[0][i] = tokens[i].intValue;
            }

            // Create speaker_ids tensor [1]
            auto speakerTensor = torch::zeros({1}, torch::kInt64);
            speakerTensor[0] = speakerId;

            // Create durs_rate tensor [1, seq_len] - all ones
            auto dursRateTensor = torch::ones({1, seqLen}, torch::kFloat32);

            // Create pitch_coefs tensor [1, seq_len] - all ones
            auto pitchCoefsTensor = torch::ones({1, seqLen}, torch::kFloat32);

            // Create empty symb_durs dict
            c10::Dict<int64_t, int64_t> symbDurs;

            // Run inference
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(sequenceTensor);
            inputs.push_back(speakerTensor);
            inputs.push_back((int64_t)sampleRate);
            inputs.push_back(symbDurs);
            inputs.push_back(dursRateTensor);
            inputs.push_back(pitchCoefsTensor);

            auto output = _model.forward(inputs);

            // Parse output: ((mag, x, y), durations)
            auto outputTuple = output.toTuple();
            auto mainOutput = outputTuple->elements()[0].toTuple();

            auto mag = mainOutput->elements()[0].toTensor().contiguous();
            auto x = mainOutput->elements()[1].toTensor().contiguous();
            auto y = mainOutput->elements()[2].toTensor().contiguous();

            // Get dimensions [1, nFreqs, nFrames]
            int nFreqs = (int)mag.size(1);
            int nFrames = (int)mag.size(2);

            // Compute complex spectrogram: mag * (x + j*y) then ISTFT
            // For simplicity, do ISTFT in C++ using the model's native ISTFT
            // Actually, let's compute audio = istft(mag * (x + 1j * y))
            // and then apply PQMF

            auto complexSpec = mag * torch::complex(x, y);

            // Use torch istft
            auto audio = torch::istft(
                complexSpec,
                /*n_fft=*/2400,
                /*hop_length=*/600,
                /*win_length=*/2400,
                /*window=*/torch::hann_window(2400),
                /*center=*/false,
                /*normalized=*/false,
                /*onesided=*/true,
                /*length=*/c10::nullopt
            );

            // Trim padding: (win_length - hop_length) / 2 from each side
            int pad = (2400 - 600) / 2;
            int audioLen = (int)audio.size(-1);
            if (audioLen > 2 * pad) {
                audio = audio.slice(-1, pad, audioLen - pad);
            }

            // Apply PQMF if needed (for 24kHz or 8kHz)
            // For now, just return the 48kHz audio - PQMF will be applied in Swift
            // if needed, or we can downsample simply

            auto audioFlat = audio.flatten().contiguous().to(torch::kFloat32);
            int totalSamples = (int)audioFlat.numel();
            float *audioData = audioFlat.data_ptr<float>();

            NSMutableArray<NSNumber *> *audioArray = [NSMutableArray arrayWithCapacity:totalSamples];
            for (int i = 0; i < totalSamples; i++) {
                [audioArray addObject:@(audioData[i])];
            }

            result = [audioArray copy];

        } @catch (NSException *exception) {
            NSLog(@"[SileroModelBridge] Inference failed: %@", exception.reason);
            result = nil;
        }
    });

    return result;
}

@end
