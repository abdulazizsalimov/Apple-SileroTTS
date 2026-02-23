import Foundation
import os

enum Log {
    private static let subsystem = "com.silero.tts"

    enum LogType: String {
        case synthesizer = "Synthesizer"
        case engine = "Engine"
        case app = "App"
        case settings = "Settings"
    }

    static func debug(type: LogType = .app, _ message: String) {
        let logger = os.Logger(subsystem: subsystem, category: type.rawValue)
        logger.debug("\(message)")
    }

    static func info(type: LogType = .app, _ message: String) {
        let logger = os.Logger(subsystem: subsystem, category: type.rawValue)
        logger.info("\(message)")
    }

    static func warning(type: LogType = .app, _ message: String) {
        let logger = os.Logger(subsystem: subsystem, category: type.rawValue)
        logger.warning("\(message)")
    }

    static func error(type: LogType = .app, _ message: String) {
        let logger = os.Logger(subsystem: subsystem, category: type.rawValue)
        logger.error("\(message)")
    }
}
