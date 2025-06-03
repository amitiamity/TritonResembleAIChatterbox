import numpy as np
import tritonclient.http as httpclient
import soundfile as sf

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input
text = "Hello Sapna. You have got message from your awesome husband and he loves you so much."
inputs = httpclient.InferInput("TEXT", [1], "BYTES")
inputs.set_data_from_numpy(np.array([text.encode('utf-8')]))

# Request audio output
outputs = [httpclient.InferRequestedOutput("AUDIO")]

# Send inference request
response = client.infer(model_name="chatterbox", inputs=[inputs], outputs=outputs)

# Extract audio output
audio = response.as_numpy("AUDIO").astype(np.float32)


# Assuming this is your audio from Triton
if len(audio.shape) > 1:
    audio = audio.squeeze()  # Fixes (1, N) to (N,)

# Make sure it's float32
audio = audio.astype(np.float32)


# Save as clean WAV file
sf.write("output.wav", audio, samplerate=24000)
print("✅ Saved: output.wav")
