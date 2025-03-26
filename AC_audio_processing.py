import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.signal
import librosa.display

def compute_mel_spectrogram(audio, sr, n_mels=40):
    """Compute the Mel spectrogram and return log-scaled values."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    return log_mel_spec

def distort_audio(audio, distortion_type):
    """Apply random distortion to the audio."""
    if distortion_type == "noise":
        noise = np.random.normal(0, 0.02, audio.shape)  # Gaussian noise
        return audio + noise
    elif distortion_type == "quantization":
        return np.round(audio * 256) / 256  # Reducing bit depth
    elif distortion_type == "lowpass":
        b, a = scipy.signal.butter(4, 0.3, 'low')  # Lowpass filter
        return scipy.signal.filtfilt(b, a, audio)
    return audio  # If no distortion is applied

def compute_masked_psycho_acoustic_loss(mel_spec1, mel_spec2, masking_threshold=3.0):
    """Compute Psycho-Acoustic Loss per sub-band with masking."""
    min_time_steps = min(mel_spec1.shape[1], mel_spec2.shape[1])
    mel_spec1 = mel_spec1[:, :min_time_steps]
    mel_spec2 = mel_spec2[:, :min_time_steps]

    # Compute sub-band losses
    sub_band_losses = np.mean(np.abs(mel_spec1 - mel_spec2), axis=1)

    # Apply perceptual masking
    masked_losses = np.where(sub_band_losses >= masking_threshold, sub_band_losses, 0)

    # Final PAL value (average of unmasked sub-bands)
    valid_losses = masked_losses[masked_losses > 0]
    final_pal_value = np.mean(valid_losses) if len(valid_losses) > 0 else 0

    return masked_losses, final_pal_value, np.abs(mel_spec1 - mel_spec2), mel_spec1, mel_spec2

def plot_results(sub_band_losses, mel_diff, mel_spec1, mel_spec2, distortion_type):
    """
    Plot 5 graphs:
    1. Spectrogram of Original Audio
    2. Spectrogram of Distorted Audio (Now using the same colormap)
    3. Spectrogram Difference Heatmap
    4. Psycho-Acoustic Loss Across Sub-Bands
    5. Line Plot of PAL Differences Across Frequencies
    """
    plt.figure(figsize=(14, 10))

    # 🔹 Subplot 1: Spectrogram of Original Audio
    plt.subplot(3, 2, 1)
    librosa.display.specshow(mel_spec1, cmap="magma", x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Original Spectrogram")

    # 🔹 Subplot 2: Spectrogram of Distorted Audio (Now using the same colormap as original)
    plt.subplot(3, 2, 2)
    librosa.display.specshow(mel_spec2, cmap="magma", x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Distorted Spectrogram ({distortion_type})")

    # 🔹 Subplot 3: Spectrogram Difference Heatmap
    plt.subplot(3, 2, 3)
    sns.heatmap(mel_diff, cmap="coolwarm", xticklabels=False, yticklabels=False)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bands")
    plt.title("Spectrogram Difference Heatmap")3

    # 🔹 Subplot 4: Psycho-Acoustic Loss per Sub-Band
    plt.subplot(3, 2, 4)
    plt.bar(np.arange(len(sub_band_losses)), sub_band_losses, alpha=0.6, color='skyblue')
    plt.xlabel("Sub-Band Index")
    plt.ylabel("Psycho-Acoustic Loss")
    plt.title("Psycho-Acoustic Loss Across Sub-Bands")

    # 🔹 Subplot 5: Line Plot Showing Frequency Differences Over Time
    plt.subplot(3, 1, 3)
    plt.plot(np.mean(mel_diff, axis=1), color='orange', linestyle='-', marker='o', label='Difference')
    plt.plot(np.mean(mel_spec1, axis=1), color='blue', linestyle='--', marker='x', label='Original')
    plt.plot(np.mean(mel_spec2, axis=1), color='green', linestyle='-.', marker='^', label='Distorted')
    plt.xlabel("Frequency Band")
    plt.ylabel("Mean Difference")
    plt.title("Average Spectral Difference Across Frequencies")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(audio_file):
    """Main function to distort an audio file and compute Psycho-Acoustic Loss."""
    # Load original audio file
    audio, sr = librosa.load(audio_file, sr=None)

    # Resample to 16 kHz
    target_sr = 16000
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Select a random distortion
    distortion_type = random.choice(["noise", "quantization", "lowpass"])
    distorted_audio = distort_audio(audio, distortion_type)

    # Compute Mel spectrograms for original and distorted audio
    mel_spec1 = compute_mel_spectrogram(audio, target_sr)
    mel_spec2 = compute_mel_spectrogram(distorted_audio, target_sr)

    # Compute Psycho-Acoustic Loss and Spectrogram Differences
    sub_band_losses, final_pal_value, mel_diff, mel_spec1, mel_spec2 = compute_masked_psycho_acoustic_loss(mel_spec1, mel_spec2)

    # Print results
    print(f"Psycho-Acoustic Loss for Each Sub-Band (with masking): {sub_band_losses}")
    print(f"Final Weighted Psycho-Acoustic Loss (Single Value): {final_pal_value:.4f}")
    print(f"Applied Distortion: {distortion_type}")

    # Plot results
    plot_results(sub_band_losses, mel_diff, mel_spec1, mel_spec2, distortion_type)

# Example usage
if __name__ == "__main__":
    audio_file = r"C:\Users\mehed\OneDrive\Desktop\RME\Sem1\AC\Project\SampleAudio\fantasy-orchestra.wav"
    main(audio_file)
