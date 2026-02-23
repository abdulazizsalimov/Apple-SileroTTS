import Foundation

/// Swift-friendly extension for SileroModelBridge
extension SileroModelBridge {

    /// Load model from path (Swift-friendly wrapper)
    func loadModel(atPath path: String) -> Bool {
        return loadModelAtPath(path)
    }

    /// Synthesize speech from token IDs
    /// - Parameters:
    ///   - tokens: Array of token IDs
    ///   - speakerId: Speaker index (0-4)
    ///   - sampleRate: Target sample rate
    /// - Returns: Float array of audio samples
    func synthesize(tokens: [Int], speakerId: Int, sampleRate: Int) -> [Float]? {
        let nsTokens = tokens.map { NSNumber(value: $0) }
        guard let nsResult = synthesize(
            withTokens: nsTokens,
            speakerId: Int32(speakerId),
            sampleRate: Int32(sampleRate)
        ) else {
            return nil
        }
        return nsResult.map { $0.floatValue }
    }
}
