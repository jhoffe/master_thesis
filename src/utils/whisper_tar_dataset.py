import io
import os

from nemo.collections.asr.data.audio_to_text import TarredAudioToCharDataset
import torch
from transformers.models.whisper.processing_whisper import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")


class WhisperTarDataset(TarredAudioToCharDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info."""
        audio_bytes, audio_filename, offset_id = tup

        # Grab manifest entry from self.manifest_preprocessor.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))

        manifest_idx = self.manifest_processor.collection.mapping[file_id][offset_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream,
            offset=offset,
            duration=manifest_entry.duration,
            trim=self.trim,
            orig_sr=manifest_entry.orig_sr,
        )
        audio_filestream.close()

        inputs = self.processor(
            features,
            sampling_rate=16000,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer(
            manifest_entry.text_raw,
            return_tensors="pt",
        ).input_ids

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.squeeze(0),
            "transcript": manifest_entry.text_raw,
        }
