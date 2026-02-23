import Foundation

enum Constants {
    static let appGroupIdentifier = "group.com.silero.tts"
    static let audioComponentSubtype = "sltx"
    static let audioComponentManufacturer = "SlTT"
    static let audioComponentName = "SileroTTS"
    static let audioComponentDescription = "Silero Text-to-Speech"
    static let supportedVoicesFileName = "SupportedVoices.json"
    static let modelFileName = "silero_v5_ru"
    static let modelFileExtension = "jit"
    static let metadataFileName = "silero_metadata"
    static let dspParamsFileName = "silero_dsp_params"

    static let defaultSampleRate: Double = 24000
    static let outputSampleRate: Double = 24000

    enum Speaker: String, CaseIterable, Codable {
        case aidar = "aidar"
        case baya = "baya"
        case kseniya = "kseniya"
        case eugene = "eugene"
        case xenia = "xenia"

        var displayName: String {
            switch self {
            case .aidar: return "Айдар"
            case .baya: return "Бая"
            case .kseniya: return "Ксения"
            case .eugene: return "Евгений"
            case .xenia: return "Ксения (2)"
            }
        }

        var speakerId: Int {
            switch self {
            case .aidar: return 0
            case .baya: return 1
            case .kseniya: return 2
            case .eugene: return 3
            case .xenia: return 4
            }
        }

        var languageCode: String { "ru-RU" }
    }
}
