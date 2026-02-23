import Foundation
import Combine

/// Manages settings for the main app UI.
/// No App Group is used - the extension hardcodes all voices.
/// The main app only uses local UserDefaults for its own UI state.
final class SettingsStore {

    static let shared = SettingsStore()

    private let userDefaults: UserDefaults
    private let enabledVoicesKey = "EnabledVoices"

    private init() {
        userDefaults = UserDefaults.standard
    }

    // MARK: - Enabled Voices (local UI state only)

    /// Set of enabled voice identifiers (speaker raw values).
    /// This is used only for the main app UI display.
    /// The extension always provides all voices to the system.
    var enabledVoiceIDs: Set<String> {
        get {
            if let ids = userDefaults.array(forKey: enabledVoicesKey) as? [String] {
                return Set(ids)
            }
            // By default, all voices are enabled
            return Set(Constants.Speaker.allCases.map { $0.rawValue })
        }
        set {
            userDefaults.set(Array(newValue), forKey: enabledVoicesKey)
            userDefaults.synchronize()
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
}

/// Voice info structure
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
