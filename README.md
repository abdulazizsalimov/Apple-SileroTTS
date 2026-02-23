# Apple-SileroTTS

iOS-приложение для интеграции русского синтезатора речи [Silero Models](https://github.com/snakers4/silero-models) с системным TTS на iOS. Приложение работает как провайдер голосов для VoiceOver и других системных функций озвучивания текста.

## Возможности

- 5 русских голосов: Айдар, Бая, Ксения, Евгений, Ксения (v2)
- Интеграция с системным синтезатором речи iOS (AVSpeechSynthesisProvider)
- Голоса отображаются в настройках VoiceOver
- Включение/отключение отдельных голосов
- Предпрослушивание голосов в приложении
- Поддержка iOS 16.0+

## Архитектура

Проект состоит из двух целей (targets):

### SileroTTS (основное приложение)
- SwiftUI интерфейс для управления голосами
- Включение/отключение голосов
- Предпрослушивание голосов

### SileroTTSExtension (Audio Unit Extension)
- Реализует `AVSpeechSynthesisProviderAudioUnit`
- Предоставляет голоса системному синтезатору
- Выполняет синтез речи с помощью модели Silero

### Common (общий код)
- `SileroTTSEngine` — координатор синтеза речи
- `TextProcessor` — токенизация текста для модели
- `DSPProcessor` — ISTFT и PQMF постобработка аудио
- `SileroModelBridge` — Objective-C++ мост для LibTorch
- `SettingsStore` — общие настройки через App Groups
- `Constants` — константы проекта

## Требования

- macOS с Xcode 15.0+
- CocoaPods
- Python 3.8+ (для подготовки модели)
- iOS 16.0+ устройство или симулятор

## Установка и сборка

### 1. Клонирование репозитория

```bash
git clone https://github.com/abdulazizsalimov/Apple-SileroTTS.git
cd Apple-SileroTTS
```

### 2. Загрузка и подготовка модели

```bash
# Загрузка модели Silero v5 Russian
chmod +x Scripts/download_model.sh
./Scripts/download_model.sh

# Трассировка модели для мобильного развёртывания
pip install torch torchaudio
python3 Scripts/trace_model.py
```

### 3. Установка зависимостей

```bash
pod install
```

### 4. Добавление модели в проект

1. Откройте `SileroTTS.xcworkspace` в Xcode
2. Перетащите файл `Models/silero_v5_ru.ptl` в проект
3. Убедитесь, что модель добавлена в оба таргета (SileroTTS и SileroTTSExtension)

### 5. Сборка и запуск

1. Откройте `SileroTTS.xcworkspace` (не `.xcodeproj`!)
2. Выберите таргет `SileroTTS`
3. Выберите устройство или симулятор
4. Нажмите Build & Run (Cmd+R)

## Использование

### Включение голосов

1. Откройте приложение SileroTTS
2. Включите нужные голоса переключателями
3. Нажмите кнопку воспроизведения для предпрослушивания

### Настройка VoiceOver

1. Перейдите в Настройки → Универсальный доступ → VoiceOver → Речь
2. Выберите язык "Русский"
3. Включённые голоса Silero будут доступны в списке

## Структура проекта

```
Apple-SileroTTS/
├── Common/
│   ├── Constants.swift
│   ├── Engine/
│   │   ├── TextProcessor.swift
│   │   ├── DSPProcessor.swift
│   │   ├── SileroTTSEngine.swift
│   │   ├── SileroModelBridge.h
│   │   ├── SileroModelBridge.mm
│   │   ├── SileroModelBridge+Swift.swift
│   │   └── SileroTTS-Bridging-Header.h
│   ├── Settings/
│   │   └── SettingsStore.swift
│   └── Utils/
│       └── Logger.swift
├── SileroTTS/
│   ├── Sources/
│   │   ├── App/SileroTTSApp.swift
│   │   ├── Views/
│   │   │   ├── ContentView.swift
│   │   │   └── VoiceRow.swift
│   │   └── Models/VoiceManager.swift
│   ├── Resources/Assets.xcassets/
│   ├── Info.plist
│   └── SileroTTS.entitlements
├── SileroTTSExtension/
│   ├── AudioUnit/SileroAudioUnit.swift
│   ├── AudioUnitFactory/AudioUnitFactory.swift
│   ├── Info.plist
│   └── SileroTTSExtension.entitlements
├── Scripts/
│   ├── download_model.sh
│   └── trace_model.py
├── Models/              (создаётся скриптом)
├── Podfile
└── SileroTTS.xcodeproj/
```

## Голоса

| ID | Имя | Описание |
|----|-----|----------|
| aidar | Айдар | Мужской голос |
| baya | Бая | Женский голос |
| kseniya | Ксения | Женский голос |
| eugene | Евгений | Мужской голос |
| xenia | Ксения (v2) | Женский голос |

## Технические детали

- **Модель**: Silero TTS v5 Russian
- **Фреймворк**: LibTorch Mobile (PyTorch Lite)
- **Частота дискретизации**: 24000 Гц
- **Формат аудио**: PCM Float32, моно
- **Токенизация**: Посимвольная (47 символов: русский алфавит + знаки препинания)
- **Постобработка**: ISTFT (n_fft=2400, hop=600) + PQMF фильтрация

## Благодарности

- [Silero Models](https://github.com/snakers4/silero-models) — модели синтеза речи
- [Apple-RHVoice](https://github.com/nicklama/Apple-RHVoice) — референсная архитектура для iOS TTS

## Лицензия

MIT License
