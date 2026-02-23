import Foundation
import AVFoundation

/// Main TTS engine that coordinates text processing and model inference.
/// This engine wraps the Silero v5_ru model and provides a simple API for synthesis.
///
/// On iOS, the model runs via LibTorch Mobile (TorchScript JIT).
/// Text is tokenized in Swift, then passed to the model for inference.
/// Post-processing (ISTFT + PQMF) converts model output to audio samples.
final class SileroTTSEngine {

    /// Shared instance
    static let shared = SileroTTSEngine()

    /// DSP post-processor
    private let dsp = DSPProcessor()

    /// Whether the engine is initialized
    private(set) var isInitialized = false

    /// Available speakers
    let speakers = Constants.Speaker.allCases

    /// Model sample rate (native output is 48kHz, downsampled as needed)
    let nativeSampleRate: Int = 48000

    /// Output sample rate for the audio unit
    let outputSampleRate: Int = 24000

    private init() {}

    /// Initialize the engine by loading the model.
    /// Call this when the app starts or when the audio unit allocates resources.
    func initialize() {
        guard !isInitialized else { return }

        // Load the TorchScript model via LibTorch
        guard let modelPath = Bundle.main.path(
            forResource: Constants.modelFileName,
            ofType: Constants.modelFileExtension
        ) ?? Bundle(for: type(of: self)).path(
            forResource: Constants.modelFileName,
            ofType: Constants.modelFileExtension
        ) else {
            Log.error(type: .engine, "Could not find model file: \(Constants.modelFileName).\(Constants.modelFileExtension)")
            return
        }

        Log.info(type: .engine, "Loading model from: \(modelPath)")

        // NOTE: Actual LibTorch model loading is done via the Objective-C++ bridge.
        // See SileroModelBridge.h/mm for the implementation.
        let success = SileroModelBridge.shared.loadModel(atPath: modelPath)
        if success {
            isInitialized = true
            Log.info(type: .engine, "Model loaded successfully")
        } else {
            Log.error(type: .engine, "Failed to load model")
        }
    }

    /// Synthesize speech from text.
    /// - Parameters:
    ///   - text: Input text in Russian
    ///   - speaker: Voice to use
    ///   - sampleRate: Output sample rate (8000, 24000, or 48000)
    /// - Returns: Audio samples as Float32 array, or nil on failure
    func synthesize(text: String, speaker: Constants.Speaker,
                    sampleRate: Int = 24000) -> [Float]? {
        guard isInitialized else {
            Log.error(type: .engine, "Engine not initialized")
            return nil
        }

        // Split text into sentences for better processing
        let sentences = TextProcessor.splitIntoSentences(text)
        var allAudio: [Float] = []

        for sentence in sentences {
            guard let audio = synthesizeSentence(sentence, speaker: speaker,
                                                  sampleRate: sampleRate) else {
                continue
            }
            allAudio.append(contentsOf: audio)
        }

        return allAudio.isEmpty ? nil : allAudio
    }

    /// Synthesize a single sentence.
    private func synthesizeSentence(_ text: String, speaker: Constants.Speaker,
                                     sampleRate: Int) -> [Float]? {
        // Tokenize text
        let tokens = TextProcessor.tokenize(text: text)

        guard tokens.count > 2 else {
            // Only SOS and EOS tokens - empty text
            return nil
        }

        Log.debug(type: .engine, "Synthesizing: '\(text)' with \(tokens.count) tokens, speaker: \(speaker.rawValue)")

        // Run model inference via the bridge
        guard let result = SileroModelBridge.shared.synthesize(
            tokens: tokens,
            speakerId: speaker.speakerId,
            sampleRate: sampleRate
        ) else {
            Log.error(type: .engine, "Model inference failed")
            return nil
        }

        return result
    }

    /// Deinitialize the engine and free resources.
    func deinitialize() {
        SileroModelBridge.shared.unloadModel()
        isInitialized = false
        Log.info(type: .engine, "Engine deinitialized")
    }
}
