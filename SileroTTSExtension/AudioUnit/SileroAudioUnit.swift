import AVFoundation
import Accelerate

/// Audio Unit Extension that provides Silero TTS voices to the iOS system.
/// This makes voices appear in VoiceOver and other system TTS consumers.
public final class SileroAudioUnit: AVSpeechSynthesisProviderAudioUnit {

    private var outputBus: AUAudioUnitBus
    private var _outputBusses: AUAudioUnitBusArray!

    private var request: AVSpeechSynthesisProviderRequest?
    private var format: AVAudioFormat
    private let sampleRate: Double = 24000.0

    private var outputData: [Float] = []
    private var outputOffset: Int = 0
    private let outputDataQueue = DispatchQueue(label: "com.silero.tts.audiounit.output", qos: .userInteractive)

    @objc override init(componentDescription: AudioComponentDescription,
                        options: AudioComponentInstantiationOptions) throws {
        self.format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: true
        )!

        outputBus = try AUAudioUnitBus(format: self.format)
        try super.init(componentDescription: componentDescription, options: options)
        _outputBusses = AUAudioUnitBusArray(
            audioUnit: self,
            busType: .output,
            busses: [outputBus]
        )
    }

    public override var outputBusses: AUAudioUnitBusArray {
        return _outputBusses
    }

    public override func allocateRenderResources() throws {
        try super.allocateRenderResources()
        SileroTTSEngine.shared.initialize()
    }

    public override func deallocateRenderResources() {
        super.deallocateRenderResources()
        cleanUp()
        SileroTTSEngine.shared.deinitialize()
    }

    // MARK: - Speech Voices

    /// Returns the list of available voices to the system.
    /// These appear in Settings > Accessibility > VoiceOver > Speech > Voice.
    public override var speechVoices: [AVSpeechSynthesisProviderVoice] {
        get {
            let voices = supportedVoices
            Log.debug(type: .synthesizer, "Returning \(voices.count) voices to system")
            return voices
        }
        set {
            Log.debug(type: .synthesizer, "System attempted to set voices: \(newValue.count)")
        }
    }

    private var supportedVoices: [AVSpeechSynthesisProviderVoice] {
        // All voices are always available - no App Group or shared settings needed.
        // The model is bundled with the extension.
        return Constants.Speaker.allCases.map { speaker in
            AVSpeechSynthesisProviderVoice(
                name: speaker.rawValue,
                identifier: "com.silero.tts.\(speaker.rawValue)",
                primaryLanguages: [speaker.languageCode],
                supportedLanguages: [speaker.languageCode]
            )
        }
    }

    // MARK: - Speech Synthesis

    public override func synthesizeSpeechRequest(_ speechRequest: AVSpeechSynthesisProviderRequest) {
        cancelSpeechRequest()
        self.request = speechRequest

        let text = extractText(from: speechRequest)
        let voiceName = speechRequest.voice.name

        Log.debug(type: .synthesizer, "Synthesis request: voice=\(voiceName), text='\(text)'")

        // Find the speaker matching the voice
        let speaker = Constants.Speaker.allCases.first { $0.rawValue == voiceName }
            ?? .xenia

        // Synthesize on a background queue to not block the audio thread
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            guard let self = self else { return }

            guard let audioSamples = SileroTTSEngine.shared.synthesize(
                text: text,
                speaker: speaker,
                sampleRate: Int(self.sampleRate)
            ) else {
                Log.error(type: .synthesizer, "Synthesis failed for text: '\(text)'")
                return
            }

            self.outputDataQueue.sync {
                self.outputData = audioSamples
                self.outputOffset = 0
            }

            Log.debug(type: .synthesizer, "Synthesized \(audioSamples.count) samples")
        }
    }

    public override func cancelSpeechRequest() {
        cleanUp()
        Log.debug(type: .synthesizer, "Speech request cancelled")
    }

    // MARK: - Rendering

    public override var internalRenderBlock: AUInternalRenderBlock {
        return { [weak self] actionFlags, timestamp, frameCount, outputBusNumber,
                  outputAudioBufferList, renderEvents, renderPull in
            guard let self = self else {
                actionFlags.pointee = .unitRenderAction_PostRenderError
                return kAudioComponentErr_InstanceInvalidated
            }
            return self.performRender(
                actionFlags: actionFlags,
                frameCount: frameCount,
                outputAudioBufferList: outputAudioBufferList
            )
        }
    }

    private func performRender(
        actionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
        frameCount: AUAudioFrameCount,
        outputAudioBufferList: UnsafeMutablePointer<AudioBufferList>
    ) -> AUAudioUnitStatus {
        let intFrameCount = Int(frameCount)

        var dataCount = 0
        var isComplete = false

        outputDataQueue.sync {
            dataCount = outputData.count
            isComplete = outputOffset >= dataCount && dataCount > 0
        }

        if isComplete {
            Log.debug(type: .synthesizer, "Rendering complete")
            actionFlags.pointee = .offlineUnitRenderAction_Complete
            cleanUp()
            return noErr
        }

        // If no data yet, wait a bit (synthesis might still be in progress)
        if dataCount == 0 {
            // Fill with silence
            outputAudioBufferList.pointee.mNumberBuffers = 1
            var buffer = UnsafeMutableAudioBufferListPointer(outputAudioBufferList)[0]
            let frames = buffer.mData!.assumingMemoryBound(to: Float32.self)
            frames.update(repeating: 0, count: intFrameCount)
            buffer.mDataByteSize = UInt32(intFrameCount * MemoryLayout<Float32>.size)
            buffer.mNumberChannels = 1
            actionFlags.pointee = .offlineUnitRenderAction_Render
            return noErr
        }

        var available = 0
        var framesToCopy: [Float] = []

        outputDataQueue.sync {
            available = min(dataCount - outputOffset, intFrameCount)
            if available > 0 && outputOffset >= 0 && outputOffset + available <= outputData.count {
                framesToCopy = Array(outputData[outputOffset..<(outputOffset + available)])
            }
        }

        outputAudioBufferList.pointee.mNumberBuffers = 1
        var buffer = UnsafeMutableAudioBufferListPointer(outputAudioBufferList)[0]
        let frames = buffer.mData!.assumingMemoryBound(to: Float32.self)

        // Fill buffer
        frames.update(repeating: 0, count: intFrameCount)
        for (i, sample) in framesToCopy.enumerated() {
            frames[i] = sample
        }

        buffer.mDataByteSize = UInt32(available * MemoryLayout<Float32>.size)
        buffer.mNumberChannels = 1

        outputDataQueue.sync {
            outputOffset += available
        }

        actionFlags.pointee = .offlineUnitRenderAction_Render
        return noErr
    }

    public override var canProcessInPlace: Bool {
        return true
    }

    // MARK: - Helpers

    private func cleanUp() {
        request = nil
        outputDataQueue.sync {
            outputData = []
            outputOffset = 0
        }
    }

    /// Extract plain text from the speech request's SSML representation.
    private func extractText(from request: AVSpeechSynthesisProviderRequest) -> String {
        let ssml = request.ssmlRepresentation

        // Strip SSML tags to get plain text
        var text = ssml
        // Remove all XML tags
        text = text.replacingOccurrences(
            of: "<[^>]+>",
            with: "",
            options: .regularExpression
        )
        // Decode common XML entities
        text = text.replacingOccurrences(of: "&amp;", with: "&")
        text = text.replacingOccurrences(of: "&lt;", with: "<")
        text = text.replacingOccurrences(of: "&gt;", with: ">")
        text = text.replacingOccurrences(of: "&quot;", with: "\"")
        text = text.replacingOccurrences(of: "&apos;", with: "'")

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
