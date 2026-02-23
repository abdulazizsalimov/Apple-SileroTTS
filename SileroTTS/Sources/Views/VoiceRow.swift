import SwiftUI

struct VoiceRow: View {
    let voice: VoiceItem
    let onToggle: () -> Void
    let onPreview: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(voice.displayName)
                    .font(.body)
                Text(voice.name)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Button(action: onPreview) {
                Image(systemName: "play.circle")
                    .font(.title2)
                    .foregroundColor(.accentColor)
            }
            .buttonStyle(BorderlessButtonStyle())

            Toggle("", isOn: Binding(
                get: { voice.isEnabled },
                set: { _ in onToggle() }
            ))
            .labelsHidden()
        }
        .padding(.vertical, 4)
    }
}
