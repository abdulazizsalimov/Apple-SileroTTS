import Foundation
import Combine

/// Manages shared settings between the main app and the TTS extension.
/// Uses UserDefaults with an App Group for cross-process communication.
final class SettingsStore {

    static let shared = SettingsStore()

    private let userDefaults: UserDefaults?
    private let enabledVoicesKey = "EnabledVoices"
    private let supportedVoicesKey = "SupportedVoices"

    private init() {
        userDefaults = UserDefaults(suiteName: Constants.appGroupIdentifier)
        // Ensure supported voices are populated on first access
        // so the extension can find them even without launching the main app
        if supportedVoices == nil {
            updateSupportedVoices()
        }
    }

    // MARK: - Enabled Voices

    /// Set of enabled voice identifiers (speaker raw values)
    var enabledVoiceIDs: Set<String> {
        get {
            if let ids = userDefaults?.array(forKey: enabledVoicesKey) as? [String] {
                return Set(ids)
            }
            // By default, all voices are enabled
            return Set(Constants.Speaker.allCases.map { $0.rawValue })
        }
        set {
            userDefaults?.set(Array(newValue), forKey: enabledVoicesKey)
            userDefaults?.synchronize()
        }
    }

    /// Check if a voice is enabled
    func isVoiceEnabled(_ speaker: Constants.Speaker) -> Bool {
        return enabledVoiceIDs.contains(speaker.rawValue)
    }

    /// Enable or disable a voice
    func setVoiceEnabled(_ speaker: Constants.Speaker, enabled: Bool) {
        var ids = enabledVoiceIDs
        if enabled {
            ids.insert(speaker.rawValue)
        } else {
            ids.remove(speaker.rawValue)
        }
        enabledVoiceIDs = ids
    }

    // MARK: - Supported Voices for Extension

    /// The list of voices that should appear in the system voice picker.
    /// Stored as JSON data.
    var supportedVoices: [SileroVoiceInfo]? {
        get {
            guard let data = userDefaults?.data(forKey: supportedVoicesKey) else {
                return nil
            }
            return try? JSONDecoder().decode([SileroVoiceInfo].self, from: data)
        }
        set {
            if let newValue = newValue,
               let data = try? JSONEncoder().encode(newValue) {
                userDefaults?.set(data, forKey: supportedVoicesKey)
            } else {
                userDefaults?.removeObject(forKey: supportedVoicesKey)
            }
            userDefaults?.synchronize()
        }
    }

    /// Update the supported voices list based on enabled voices.
    func updateSupportedVoices() {
        let enabled = enabledVoiceIDs
        let voices = Constants.Speaker.allCases
            .filter { enabled.contains($0.rawValue) }
            .map { SileroVoiceInfo(speaker: $0) }
        supportedVoices = voices
    }
}

/// Serializable voice info for sharing between app and extension.
struct SileroVoiceInfo: Codable, Hashable {
    let identifier: String
    let name: String
    let displayName: String
    let languageCode: String
    let speakerId: Int

    init(speaker: Constants.Speaker) {
        self.identifier = "com.silero.tts.\(speaker.rawValue)"
        self.name = speaker.rawValue
        self.displayName = speaker.displayName
        self.languageCode = speaker.languageCode
        self.speakerId = speaker.speakerId
    }
}
