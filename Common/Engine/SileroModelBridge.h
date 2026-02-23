#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Objective-C++ bridge for LibTorch model inference.
/// This class handles loading the TorchScript JIT model and running inference.
@interface SileroModelBridge : NSObject

/// Shared instance
@property (class, readonly) SileroModelBridge *shared;

/// Whether the model is loaded
@property (nonatomic, readonly) BOOL isModelLoaded;

/// Load the TorchScript model from the given path.
/// @param path Path to the .jit model file
/// @return YES if successful
- (BOOL)loadModelAtPath:(NSString *)path;

/// Unload the model and free resources.
- (void)unloadModel;

/// Run TTS inference.
/// @param tokens Array of token IDs (NSNumber<int>)
/// @param speakerId Speaker ID (0-4)
/// @param sampleRate Target sample rate (8000, 24000, or 48000)
/// @return Array of Float32 audio samples (NSNumber<float>), or nil on failure
- (nullable NSArray<NSNumber *> *)synthesizeWithTokens:(NSArray<NSNumber *> *)tokens
                                             speakerId:(int)speakerId
                                            sampleRate:(int)sampleRate;

@end

NS_ASSUME_NONNULL_END
