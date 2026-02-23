import Foundation
import Accelerate

/// Performs ISTFT and PQMF post-processing to convert model output to audio samples.
/// The Silero model outputs (magnitude, real, imaginary) tensors that need ISTFT + PQMF
/// to produce the final waveform.
final class DSPProcessor {

    // ISTFT parameters matching the Silero v5_ru model
    let nFFT: Int = 2400
    let hopLength: Int = 600
    let winLength: Int = 2400

    /// Hann window for ISTFT
    private let hannWindow: [Float]

    /// PQMF filter coefficients for 2x downsampling (48kHz → 24kHz)
    private let pqmf2H: [[Float]]
    private let pqmf2N: Int = 2
    private let pqmf2Taps: Int = 62

    /// PQMF filter coefficients for 6x downsampling (48kHz → 8kHz)
    private let pqmf6H: [[Float]]
    private let pqmf6N: Int = 6
    private let pqmf6Taps: Int = 62

    init() {
        // Generate Hann window
        var window = [Float](repeating: 0, count: winLength)
        vDSP_hann_window(&window, vDSP_Length(winLength), Int32(vDSP_HANN_NORM))
        self.hannWindow = window

        // Generate PQMF-2 filters
        self.pqmf2H = DSPProcessor.generatePQMFFilters(N: 2, taps: 62, cutoff: 0.25, beta: 10.0)

        // Generate PQMF-6 filters
        self.pqmf6H = DSPProcessor.generatePQMFFilters(N: 6, taps: 62, cutoff: 0.12, beta: 9.0)
    }

    /// Generate PQMF analysis filter bank coefficients.
    private static func generatePQMFFilters(N: Int, taps: Int, cutoff: Double, beta: Double) -> [[Float]] {
        // Generate Kaiser-windowed FIR lowpass filter
        let filterLen = taps + 1
        let qmf = kaiserFIR(numTaps: filterLen, cutoff: cutoff, beta: beta)

        var H = [[Float]](repeating: [Float](repeating: 0, count: filterLen), count: N)

        for k in 0..<N {
            for n in 0..<filterLen {
                let constantFactor = Double(2 * k + 1) * .pi / Double(2 * N) * (Double(n) - Double(taps - 1) / 2.0)
                let phase = (k % 2 == 0 ? 1.0 : -1.0) * .pi / 4.0
                H[k][n] = Float(2.0 * qmf[n] * cos(constantFactor + phase))
            }
        }

        return H
    }

    /// Generate a Kaiser-windowed FIR lowpass filter.
    private static func kaiserFIR(numTaps: Int, cutoff: Double, beta: Double) -> [Double] {
        let M = numTaps - 1
        var h = [Double](repeating: 0, count: numTaps)

        // Generate sinc filter
        for n in 0..<numTaps {
            let x = Double(n) - Double(M) / 2.0
            if abs(x) < 1e-10 {
                h[n] = 2.0 * cutoff
            } else {
                h[n] = sin(2.0 * .pi * cutoff * x) / (.pi * x)
            }
        }

        // Apply Kaiser window
        let kaiserWin = kaiserWindow(length: numTaps, beta: beta)
        for n in 0..<numTaps {
            h[n] *= kaiserWin[n]
        }

        return h
    }

    /// Generate a Kaiser window.
    private static func kaiserWindow(length: Int, beta: Double) -> [Double] {
        var window = [Double](repeating: 0, count: length)
        let denom = besselI0(beta)

        for n in 0..<length {
            let arg = beta * sqrt(1.0 - pow(2.0 * Double(n) / Double(length - 1) - 1.0, 2.0))
            window[n] = besselI0(arg) / denom
        }

        return window
    }

    /// Modified Bessel function of the first kind, order 0.
    private static func besselI0(_ x: Double) -> Double {
        var sum = 1.0
        var term = 1.0
        for k in 1...50 {
            term *= (x / (2.0 * Double(k))) * (x / (2.0 * Double(k)))
            sum += term
            if term < 1e-12 { break }
        }
        return sum
    }

    /// Perform ISTFT on complex spectrogram to produce time-domain audio.
    /// - Parameters:
    ///   - magnitude: Magnitude tensor [1, nFreqs, nFrames]
    ///   - real: Real part tensor [1, nFreqs, nFrames]
    ///   - imaginary: Imaginary part tensor [1, nFreqs, nFrames]
    /// - Returns: Audio samples at 48kHz
    func istft(magnitude: [Float], real: [Float], imaginary: [Float],
               nFreqs: Int, nFrames: Int) -> [Float] {

        let pad = (winLength - hopLength) / 2
        let outputSize = (nFrames - 1) * hopLength + winLength

        var output = [Float](repeating: 0, count: outputSize)
        var windowEnvelope = [Float](repeating: 0, count: outputSize)

        // Process each frame
        for t in 0..<nFrames {
            // Build complex spectrum: mag * (real + j*imag)
            var frameReal = [Float](repeating: 0, count: nFFT)
            var frameImag = [Float](repeating: 0, count: nFFT)

            for f in 0..<nFreqs {
                let idx = f * nFrames + t
                let mag = magnitude[idx]
                let re = real[idx]
                let im = imaginary[idx]
                frameReal[f] = mag * re
                frameImag[f] = mag * im
            }

            // IRFFT: Convert from nFreqs complex to nFFT real
            var timeDomain = irfft(real: frameReal, imag: frameImag, nFFT: nFFT)

            // Apply window
            vDSP_vmul(timeDomain, 1, hannWindow, 1, &timeDomain, 1, vDSP_Length(winLength))

            // Overlap-add
            let startIdx = t * hopLength
            for i in 0..<winLength {
                output[startIdx + i] += timeDomain[i]
            }

            // Window envelope
            var windowSq = [Float](repeating: 0, count: winLength)
            vDSP_vsq(hannWindow, 1, &windowSq, 1, vDSP_Length(winLength))
            for i in 0..<winLength {
                windowEnvelope[startIdx + i] += windowSq[i]
            }
        }

        // Normalize by window envelope and trim padding
        let trimmedLen = outputSize - 2 * pad
        var result = [Float](repeating: 0, count: trimmedLen)

        for i in 0..<trimmedLen {
            let env = windowEnvelope[i + pad]
            if env > 1e-11 {
                result[i] = output[i + pad] / env
            }
        }

        return result
    }

    /// Inverse real FFT: convert nFreqs complex coefficients to nFFT real samples.
    private func irfft(real: [Float], imag: [Float], nFFT: Int) -> [Float] {
        let nFreqs = nFFT / 2 + 1
        var result = [Float](repeating: 0, count: nFFT)

        // Use Accelerate's vDSP for FFT
        let log2n = vDSP_Length(log2(Float(nFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            return result
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Pack into split complex format
        var realPart = [Float](repeating: 0, count: nFFT / 2)
        var imagPart = [Float](repeating: 0, count: nFFT / 2)

        // DC component
        realPart[0] = real[0]
        imagPart[0] = real[nFreqs - 1]  // Nyquist

        for i in 1..<(nFFT / 2) {
            realPart[i] = real[i]
            imagPart[i] = imag[i]
        }

        var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)

        // Inverse FFT
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(kFFTDirection_Inverse))

        // Unpack result
        let scale = 1.0 / Float(nFFT)
        for i in 0..<(nFFT / 2) {
            result[2 * i] = realPart[i] * scale
            result[2 * i + 1] = imagPart[i] * scale
        }

        return result
    }

    /// Apply PQMF analysis filter bank for downsampling.
    /// - Parameters:
    ///   - audio: Input audio [samples]
    ///   - sampleRate: Target sample rate
    /// - Returns: Downsampled audio
    func applyPQMF(audio: [Float], targetSampleRate: Int) -> [Float] {
        if targetSampleRate == 48000 {
            return audio
        }

        let H: [[Float]]
        let N: Int
        let taps: Int

        if targetSampleRate == 24000 {
            H = pqmf2H
            N = pqmf2N
            taps = pqmf2Taps
        } else if targetSampleRate == 8000 {
            H = pqmf6H
            N = pqmf6N
            taps = pqmf6Taps
        } else {
            return audio
        }

        // Apply convolution with stride N, take first channel only
        let padding = taps / 2
        let paddedLen = audio.count + 2 * padding
        var padded = [Float](repeating: 0, count: paddedLen)
        for i in 0..<audio.count {
            padded[i + padding] = audio[i]
        }

        let outputLen = (paddedLen - (taps + 1)) / N + 1
        var result = [Float](repeating: 0, count: outputLen)

        // Convolve with first filter only (channel 0)
        let filter = H[0]
        for i in 0..<outputLen {
            var sum: Float = 0
            let startIdx = i * N
            vDSP_dotpr(padded.advanced(by: startIdx), 1,
                       filter, 1, &sum, vDSP_Length(taps + 1))
            result[i] = sum
        }

        // Clamp to [-1, 1]
        var maxVal: Float = 0
        vDSP_maxmgv(result, 1, &maxVal, vDSP_Length(result.count))
        if maxVal > 1.0 {
            // Apply tanh
            var count = Int32(result.count)
            vvtanhf(&result, result, &count)
        }

        return result
    }

    /// Complete post-processing pipeline: ISTFT + PQMF
    /// - Parameters:
    ///   - magnitude: Model output magnitude [nFreqs * nFrames]
    ///   - real: Model output real [nFreqs * nFrames]
    ///   - imaginary: Model output imaginary [nFreqs * nFrames]
    ///   - nFreqs: Number of frequency bins
    ///   - nFrames: Number of time frames
    ///   - sampleRate: Target sample rate (8000, 24000, or 48000)
    /// - Returns: Audio samples as Float array
    func processModelOutput(magnitude: [Float], real: [Float], imaginary: [Float],
                            nFreqs: Int, nFrames: Int, sampleRate: Int) -> [Float] {
        let audio48k = istft(magnitude: magnitude, real: real, imaginary: imaginary,
                             nFreqs: nFreqs, nFrames: nFrames)
        return applyPQMF(audio: audio48k, targetSampleRate: sampleRate)
    }
}
