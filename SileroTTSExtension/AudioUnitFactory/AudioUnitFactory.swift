import CoreAudioKit

public class AudioUnitFactory: NSObject, AUAudioUnitFactory {
    var auAudioUnit: AUAudioUnit?

    public func beginRequest(with context: NSExtensionContext) {
    }

    @objc
    public func createAudioUnit(with componentDescription: AudioComponentDescription) throws -> AUAudioUnit {
        auAudioUnit = try SileroAudioUnit(componentDescription: componentDescription, options: [])

        guard let audioUnit = auAudioUnit as? SileroAudioUnit else {
            fatalError("Failed to create SileroAudioUnit")
        }
        return audioUnit
    }
}
