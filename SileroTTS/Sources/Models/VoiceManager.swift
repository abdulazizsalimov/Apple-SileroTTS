import Foundation
import Combine
import AVFoundation

/// Manages voice state and provides API for the main app UI.
final class VoiceManager: ObservableObject {

    /// All available voices with their enabled state
    @Published var voices: [VoiceItem] = []

    /// Current status message
    @Published var statusMessage: String = "Готово"

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

        Log.info(type: .settings,
                 "Voice \(voice.speaker.rawValue) \(voices[index].isEnabled ? "enabled" : "disabled")")
    }

    /// Enable all voices
    func enableAll() {
        for i in voices.indices {
            voices[i].isEnabled = true
            settings.setVoiceEnabled(voices[i].speaker, enabled: true)
        }
    }

    /// Disable all voices
    func disableAll() {
        for i in voices.indices {
            voices[i].isEnabled = false
            settings.setVoiceEnabled(voices[i].speaker, enabled: false)
        }
    }

    /// Preview a voice using system AVSpeechSynthesizer (no model loading needed)
    func previewVoice(_ voice: VoiceItem) {
        guard !isPlaying else { return }
        isPlaying = true

        let sampleText = "Привет! Меня зовут \(voice.speaker.displayName). Это пример синтеза речи."

        let utterance = AVSpeechUtterance(string: sampleText)
        // Try to use our Silero voice if available in the system
        let voiceIdentifier = "com.silero.tts.\(voice.speaker.rawValue)"
        if let sileroVoice = AVSpeechSynthesisVoice(identifier: voiceIdentifier) {
            utterance.voice = sileroVoice
        } else {
            // Fallback to system Russian voice
            utterance.voice = AVSpeechSynthesisVoice(language: "ru-RU")
        }
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate

        speechSynthesizer.speak(utterance)

        // Auto-stop after a reasonable timeout
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) { [weak self] in
            self?.isPlaying = false
        }
    }

    /// Stop current preview playback
    func stopPreview() {
        speechSynthesizer.stopSpeaking(at: .immediate)
        audioPlayer?.stop()
        audioPlayer = nil
        isPlaying = false
    }

    /// Notify system that our voices are available.
    /// The extension hardcodes all voices, so this just triggers a system refresh.
    func ensureVoicesRegistered() {
        AVSpeechSynthesisProviderVoice.updateSpeechVoices()
        statusMessage = "Голоса зарегистрированы"
    }

    // MARK: - Private

    private lazy var speechSynthesizer: AVSpeechSynthesizer = {
        let synth = AVSpeechSynthesizer()
        return synth
    }()
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
