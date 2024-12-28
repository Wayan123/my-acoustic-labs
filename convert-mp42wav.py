from pydub import AudioSegment
import os

# Ganti path di bawah dengan file MP4 yang ingin dikonversi
input_file = "drive-bersama-bintang.mp4"

# Memisahkan nama file dan ekstensi
base_name, _ = os.path.splitext(input_file)

# Membuat nama file output dengan ekstensi .wav
output_file = f"{base_name}.wav"

# Membaca file MP4 dan mengkonversi menjadi WAV
audio = AudioSegment.from_file(input_file, format="mp4")
audio.export(output_file, format="wav")

print("Konversi berhasil! File WAV telah dibuat:", output_file)
