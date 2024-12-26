import torch
import sounddevice as sd
import queue
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# -------------------------------------------------------------------
# 1. Inisialisasi Model Whisper
# -------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
print(f"Device set to use {device}")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Buat pipeline ASR
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
)

# Paksa bahasa Indonesia (supaya transkripsi default ke bahasa Indonesia)
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="Indonesian", task="transcribe"
)

# -------------------------------------------------------------------
# 2. Parameter Audio & VAD
# -------------------------------------------------------------------
sample_rate = 16000
block_size = 16000  # 1 detik per blok
silence_threshold = 0.01      # Ambang RMS untuk dianggap diam
required_silence_blocks = 2   # Berapa blok diam berurutan hingga dianggap selesai bicara
silence_count = 0

# Buffer untuk menampung audio selama user bicara
audio_buffer = []

# Rolling window untuk menampilkan spectrogram (5 detik)
rolling_window = np.array([], dtype=np.float32)
rolling_duration = 5  # detik

# -------------------------------------------------------------------
# 3. Setup input stream + queue
# -------------------------------------------------------------------
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# -------------------------------------------------------------------
# 4. Persiapan Matplotlib untuk spectrogram
# -------------------------------------------------------------------
plt.ion()  # mode interaktif, supaya plot bisa di-update
fig, ax = plt.subplots(figsize=(8, 4))

print("=== Mulai merekam (scrolling spectrogram). Tekan Ctrl+C untuk berhenti. ===")

try:
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=block_size
    ):
        while True:
            # Ambil 1 blok audio dari queue (1 detik)
            block_audio = q.get().flatten()

            # ---------------------------------------------------------------
            # 4a. Update rolling window (untuk menampilkan 5 detik terakhir)
            # ---------------------------------------------------------------
            rolling_window = np.concatenate((rolling_window, block_audio))
            max_len = int(rolling_duration * sample_rate)  # 5 * 16000 = 80000
            if len(rolling_window) > max_len:
                rolling_window = rolling_window[-max_len:]  # potong depan

            # ---------------------------------------------------------------
            # 4b. Hitung & plot rolling spectrogram (5 detik terakhir)
            # ---------------------------------------------------------------
            f, t, Sxx = signal.spectrogram(
                rolling_window,
                fs=sample_rate,
                nperseg=512,
                noverlap=256
            )
            # Sxx => (freq_bins, time_slices)

            ax.clear()

            # Kita definisikan "extent" agar sumbu horizontal 0..5 (detik),
            # dan sumbu vertikal sesuai range frekuensi (0..f[-1]).
            extent = [0, rolling_duration, f[0], f[-1]]

            # 10*np.log10(Sxx) => konversi ke dB
            # aspect='auto' => agar lebar x menyesuaikan, origin='lower' => frekuensi rendah di bawah.
            ax.imshow(
                10 * np.log10(Sxx),
                aspect='auto',
                origin='lower',
                cmap='magma',
                extent=extent
            )
            ax.set_ylabel("Frekuensi [Hz]")
            ax.set_xlabel("Waktu [detik] (window 5 detik)")
            ax.set_title("Spectrogram Bergeser Real-time (5 detik)")

            plt.draw()
            plt.pause(0.001)

            # ---------------------------------------------------------------
            # 5. VAD Sederhana (berbasis threshold RMS)
            # ---------------------------------------------------------------
            rms = np.sqrt(np.mean(block_audio ** 2))
            if rms < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0

            # Selama masih ada suara, blok audio dikumpulkan ke audio_buffer
            audio_buffer.append(block_audio)

            # ---------------------------------------------------------------
            # 6. Jika diam berurutan >= required_silence_blocks, transkripsi
            # ---------------------------------------------------------------
            if silence_count >= required_silence_blocks and len(audio_buffer) > 0:
                # print("=== Terdeteksi akhir bicara, mulai transkripsi... ===")

                # Gabungkan semua blok jadi satu array
                full_audio = np.concatenate(audio_buffer)

                # Siapkan input untuk pipeline
                audio_input = {
                    "array": full_audio,
                    "sampling_rate": sample_rate
                }

                # Jalankan pipeline Whisper
                result = asr_pipeline(
                    audio_input,
                    generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
                )

                # -----------------------------------------------------------
                # 7. Post-processing: Ganti "Terima kasih." => "Silakan berbicara."
                # -----------------------------------------------------------
                processed_text = result["text"]
                processed_text = processed_text.replace(
                    "Terima kasih.", "Silakan berbicara."
                )
                
                # Cetak hasil transkripsi
                print(">>>", processed_text)

                # Reset buffer bicara & counter
                audio_buffer = []
                silence_count = 0

except KeyboardInterrupt:
    print("\n=== Perekaman dihentikan oleh user. ===")
    plt.ioff()
    plt.show()
except Exception as e:
    print("Terjadi error:", str(e))
    plt.ioff()
    plt.show()
