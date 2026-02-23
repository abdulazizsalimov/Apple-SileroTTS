import SwiftUI

struct ContentView: View {
    @EnvironmentObject var voiceManager: VoiceManager

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Status bar
                HStack {
                    Text(voiceManager.statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color(.systemGroupedBackground))

                // Voice list
                List {
                    Section(header: Text("Голоса Silero")) {
                        ForEach(voiceManager.voices) { voice in
                            VoiceRow(voice: voice) {
                                voiceManager.toggleVoice(voice)
                            } onPreview: {
                                if voiceManager.isPlaying {
                                    voiceManager.stopPreview()
                                } else {
                                    voiceManager.previewVoice(voice)
                                }
                            }
                        }
                    }

                    Section(header: Text("Информация")) {
                        HStack {
                            Text("Модель")
                            Spacer()
                            Text("Silero v5 RU")
                                .foregroundColor(.secondary)
                        }
                        HStack {
                            Text("Язык")
                            Spacer()
                            Text("Русский")
                                .foregroundColor(.secondary)
                        }
                        HStack {
                            Text("Голосов")
                            Spacer()
                            Text("\(voiceManager.voices.filter { $0.isEnabled }.count) из \(voiceManager.voices.count)")
                                .foregroundColor(.secondary)
                        }
                    }

                    Section(header: Text("Инструкция")) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Как включить голоса в VoiceOver:")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            Text("1. Включите нужные голоса выше")
                                .font(.caption)
                            Text("2. Откройте Настройки → Универсальный доступ → VoiceOver → Речь")
                                .font(.caption)
                            Text("3. Выберите «Голос» и найдите голоса Silero")
                                .font(.caption)
                            Text("4. Выберите один из включённых голосов")
                                .font(.caption)
                        }
                        .padding(.vertical, 4)
                    }
                }
                .listStyle(InsetGroupedListStyle())
            }
            .navigationTitle("Silero TTS")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button("Включить все") {
                            voiceManager.enableAll()
                        }
                        Button("Выключить все") {
                            voiceManager.disableAll()
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
        .navigationViewStyle(StackNavigationViewStyle())
            .onAppear {
                voiceManager.ensureVoicesRegistered()
            }
    }
}
