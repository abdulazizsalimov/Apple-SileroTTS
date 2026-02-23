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
    /// Call this when the audio unit allocates resources (NOT from the main app).
    func initialize() {
        guard !isInitialized else { return }

        // Search for the model in multiple bundle locations
        // 1. Main bundle (for the app)
        // 2. Bundle containing this class (for the extension)
        // 3. Direct path in the bundle root
        let modelName = Constants.modelFileName
        let modelExt = Constants.modelFileExtension

        var modelPath: String?

        // Try the class bundle first (works for extensions)
        let classBundle = Bundle(for: SileroModelBridge.self)
        modelPath = classBundle.path(forResource: modelName, ofType: modelExt)

        // Try main bundle
        if modelPath == nil {
            modelPath = Bundle.main.path(forResource: modelName, ofType: modelExt)
        }

        // Try direct path in bundle root
        if modelPath == nil {
            let directPath = classBundle.bundlePath + "/" + modelName + "." + modelExt
            if FileManager.default.fileExists(atPath: directPath) {
                modelPath = directPath
            }
        }

        // Try direct path in main bundle root
        if modelPath == nil {
            let directPath = Bundle.main.bundlePath + "/" + modelName + "." + modelExt
            if FileManager.default.fileExists(atPath: directPath) {
                modelPath = directPath
            }
        }

        guard let finalPath = modelPath else {
            Log.error(type: .engine, "Could not find model file: \(modelName).\(modelExt)")
            Log.error(type: .engine, "Searched in: \(classBundle.bundlePath) and \(Bundle.main.bundlePath)")
            return
        }

        Log.info(type: .engine, "Loading model from: \(finalPath)")

        let success = SileroModelBridge.shared.loadModel(atPath: finalPath)
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
