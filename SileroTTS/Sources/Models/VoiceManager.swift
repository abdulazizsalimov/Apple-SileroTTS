import Foundation
import Combine
import AVFoundation

/// Manages voice state and provides API for the main app UI.
final class VoiceManager: ObservableObject {

    /// All available voices with their enabled state
    @Published var voices: [VoiceItem] = []

    /// Whether the TTS engine is ready
    @Published var isEngineReady: Bool = false

    /// Current status message
    @Published var statusMessage: String = "Инициализация..."

    /// Whether a preview is currently playing
    @Published var isPlaying: Bool = false

    private var audioPlayer: AVAudioPlayer?
    private let settings = SettingsStore.shared

    init() {
        loadVoices()
    }

    /// Load voice states from settings
    func loadVoices() {
        let enabledIDs = settings.enabledVoiceIDs
        voices = Constants.Speaker.allCases.map { speaker in
            VoiceItem(
                speaker: speaker,
                isEnabled: enabledIDs.contains(speaker.rawValue)
            )
        }
        statusMessage = "Готово"
    }

    /// Toggle a voice on/off
    func toggleVoice(_ voice: VoiceItem) {
        guard let index = voices.firstIndex(where: { $0.id == voice.id }) else { return }
        voices[index].isEnabled.toggle()

        settings.setVoiceEnabled(voice.speaker, enabled: voices[index].isEnabled)
        settings.updateSupportedVoices()

        Log.info(type: .settings,
                 "Voice \(voice.speaker.rawValue) \(voices[index].isEnabled ? "enabled" : "disabled")")
    }

    /// Enable all voices
    func enableAll() {
        for i in voices.indices {
            voices[i].isEnabled = true
            settings.setVoiceEnabled(voices[i].speaker, enabled: true)
        }
        settings.updateSupportedVoices()
    }

    /// Disable all voices
    func disableAll() {
        for i in voices.indices {
            voices[i].isEnabled = false
            settings.setVoiceEnabled(voices[i].speaker, enabled: false)
        }
        settings.updateSupportedVoices()
    }

    /// Preview a voice by synthesizing sample text
    func previewVoice(_ voice: VoiceItem) {
        guard !isPlaying else { return }
        isPlaying = true

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            let sampleText = "Привет! Меня зовут \(voice.speaker.displayName). Это пример синтеза речи."

            guard let audioSamples = SileroTTSEngine.shared.synthesize(
                text: sampleText,
                speaker: voice.speaker,
                sampleRate: Int(Constants.outputSampleRate)
            ) else {
                DispatchQueue.main.async {
                    self.isPlaying = false
                    self.statusMessage = "Ошибка синтеза"
                }
                return
            }

            // Convert samples to WAV data for playback
            let wavData = self.samplesToWAV(
                samples: audioSamples,
                sampleRate: Int(Constants.outputSampleRate)
            )

            DispatchQueue.main.async {
                do {
                    self.audioPlayer = try AVAudioPlayer(data: wavData)
                    self.audioPlayer?.delegate = nil
                    self.audioPlayer?.play()

                    // Auto-stop after duration
                    let duration = Double(audioSamples.count) / Constants.outputSampleRate
                    DispatchQueue.main.asyncAfter(deadline: .now() + duration + 0.5) {
                        self.isPlaying = false
                    }
                } catch {
                    self.isPlaying = false
                    self.statusMessage = "Ошибка воспроизведения: \(error.localizedDescription)"
                }
            }
        }
    }

    /// Stop current preview playback
    func stopPreview() {
        audioPlayer?.stop()
        audioPlayer = nil
        isPlaying = false
    }

    /// Initialize the TTS engine
    func initializeEngine() {
        statusMessage = "Загрузка модели..."
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            SileroTTSEngine.shared.initialize()
            DispatchQueue.main.async {
                self?.isEngineReady = SileroTTSEngine.shared.isInitialized
                self?.statusMessage = SileroTTSEngine.shared.isInitialized
                    ? "Модель загружена" : "Ошибка загрузки модели"
            }
        }
    }

    // MARK: - WAV Encoding

    /// Convert Float32 audio samples to WAV data
    private func samplesToWAV(samples: [Float], sampleRate: Int) -> Data {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = bitsPerSample / 8
        let dataSize = UInt32(samples.count * Int(bytesPerSample))

        var data = Data()

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        var chunkSize = UInt32(36 + dataSize)
        data.append(Data(bytes: &chunkSize, count: 4))
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        var fmtSize: UInt32 = 16
        data.append(Data(bytes: &fmtSize, count: 4))
        var audioFormat: UInt16 = 1 // PCM
        data.append(Data(bytes: &audioFormat, count: 2))
        var channels = numChannels
        data.append(Data(bytes: &channels, count: 2))
        var rate = UInt32(sampleRate)
        data.append(Data(bytes: &rate, count: 4))
        var byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bytesPerSample)
        data.append(Data(bytes: &byteRate, count: 4))
        var blockAlign = numChannels * bytesPerSample
        data.append(Data(bytes: &blockAlign, count: 2))
        var bps = bitsPerSample
        data.append(Data(bytes: &bps, count: 2))

        // data chunk
        data.append(contentsOf: "data".utf8)
        var dSize = dataSize
        data.append(Data(bytes: &dSize, count: 4))

        // Convert float samples to int16
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            var int16Sample = Int16(clamped * 32767.0)
            data.append(Data(bytes: &int16Sample, count: 2))
        }

        return data
    }
}

/// A voice item for the UI
struct VoiceItem: Identifiable {
    let id: String
    let speaker: Constants.Speaker
    var isEnabled: Bool

    init(speaker: Constants.Speaker, isEnabled: Bool) {
        self.id = speaker.rawValue
        self.speaker = speaker
        self.isEnabled = isEnabled
    }

    var displayName: String { speaker.displayName }
    var name: String { speaker.rawValue }
    var languageCode: String { speaker.languageCode }
}
