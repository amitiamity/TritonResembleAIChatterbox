import numpy as np
import triton_python_backend_utils as pb_utils
from chatterbox.tts import ChatterboxTTS
import torchaudio

class TritonPythonModel:
    def initialize(self, args):
        self.model = ChatterboxTTS.from_local(ckpt_dir="/app/models/weights", device="cuda")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            input_text = input_tensor.as_numpy()[0].decode('utf-8')

            # Generate speech
            wav = self.model.generate(input_text)
            wav = self.model.generate(input_text)
         
            audio_array = wav.numpy()

            # Create output tensor
            output_tensor = pb_utils.Tensor("AUDIO", audio_array)
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)
        return responses
