import SwiftUI

@main
struct SileroTTSApp: App {
    @StateObject private var voiceManager = VoiceManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(voiceManager)
        }
    }
}
