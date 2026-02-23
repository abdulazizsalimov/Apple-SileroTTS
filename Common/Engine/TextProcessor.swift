import Foundation

/// Handles text preprocessing for Silero TTS models.
/// Converts input text to token IDs that the model expects.
final class TextProcessor {

    /// Character-to-ID mapping matching the Silero v5_ru model
    static let symbolToId: [Character: Int] = [
        "_": 0, "~": 1, "|": 2, "!": 3, "+": 4, ",": 5,
        "-": 6, ".": 7, ":": 8, ";": 9, "?": 10,
        "а": 11, "б": 12, "в": 13, "г": 14, "д": 15,
        "е": 16, "ж": 17, "з": 18, "и": 19, "й": 20,
        "к": 21, "л": 22, "м": 23, "н": 24, "о": 25,
        "п": 26, "р": 27, "с": 28, "т": 29, "у": 30,
        "ф": 31, "х": 32, "ц": 33, "ч": 34, "ш": 35,
        "щ": 36, "ъ": 37, "ы": 38, "ь": 39, "э": 40,
        "ю": 41, "я": 42, "ё": 43, "–": 44, "…": 45,
        " ": 46
    ]

    /// Start-of-sequence token
    static let sosToken: Int = 2  // "|"
    /// End-of-sequence token
    static let eosToken: Int = 1  // "~"
    /// Accent/stress marker token
    static let accentToken: Int = 4  // "+"

    /// Allowed characters (everything in symbolToId except special tokens)
    static let allowedCharacters: Set<Character> = Set(symbolToId.keys)

    /// Preprocess and tokenize text for the Silero model.
    /// - Parameter text: Input text in Russian
    /// - Returns: Array of token IDs ready for model input
    static func tokenize(text: String) -> [Int] {
        let cleaned = preprocessText(text)
        var tokens: [Int] = [sosToken]

        for char in cleaned {
            if let tokenId = symbolToId[char] {
                tokens.append(tokenId)
            }
            // Skip unknown characters
        }

        tokens.append(eosToken)
        return tokens
    }

    /// Clean and normalize text for the model.
    /// - Parameter text: Raw input text
    /// - Returns: Cleaned text ready for tokenization
    static func preprocessText(_ text: String) -> String {
        var result = text.lowercased()

        // Replace various dash types with standard dash
        result = result.replacingOccurrences(of: "—", with: "–")
        result = result.replacingOccurrences(of: "‑", with: "-")

        // Replace multiple spaces with single space
        result = result.replacingOccurrences(
            of: "\\s+",
            with: " ",
            options: .regularExpression
        )

        // Remove characters not in our symbol set
        result = String(result.filter { allowedCharacters.contains($0) })

        result = result.trimmingCharacters(in: .whitespaces)

        return result
    }

    /// Split long text into sentences for processing.
    /// The model works best with shorter segments.
    /// - Parameter text: Input text
    /// - Returns: Array of sentence strings
    static func splitIntoSentences(_ text: String) -> [String] {
        let delimiters = CharacterSet(charactersIn: ".!?;")
        var sentences: [String] = []
        var current = ""

        for char in text {
            current.append(char)
            if let scalar = char.unicodeScalars.first, delimiters.contains(scalar) {
                let trimmed = current.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty {
                    sentences.append(trimmed)
                }
                current = ""
            }
        }

        let remaining = current.trimmingCharacters(in: .whitespaces)
        if !remaining.isEmpty {
            sentences.append(remaining)
        }

        return sentences
    }
}
