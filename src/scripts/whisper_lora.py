from utils.whisper_tar_dataset import WhisperTarDataset
import torch
import os

manifest_filepath = (
    "../datasets/processed/CoRal-project_coral-v2_read-aloud/train/tarred_audio_manifest.json"
)
audio_tar_filepaths = (
    "../datasets/processed/CoRal-project_coral-v2_read-aloud/train/audio__OP_0..295_CL_.tar"
)

dataset = WhisperTarDataset(
    audio_tar_filepaths=audio_tar_filepaths,
    manifest_filepath=manifest_filepath,
    labels=[],
    sample_rate=16000,
)
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for sequence-to-sequence speech tasks using Whisper.

    This collator dynamically pads both the input audio features and the target text tokens
    to the maximum length in a batch, making it compatible with variable-length input/output sequences.

    Attributes:
        processor (Any): A Hugging Face `WhisperProcessor` that includes both a feature extractor
                         for audio and a tokenizer for text.

    Methods:
        __call__(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            Pads and collates a batch of audio-text pairs for model input.
    """

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Pads input audio features and target text labels for a batch of samples.

        Args:
            features (List[Dict]): Each item in the list is a dictionary with:
                - 'input_features': Audio features (from spectrogram extraction)
                - 'labels': Tokenized text labels

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_features': Padded audio features
                - 'labels': Padded and masked labels (with padding tokens replaced by -100)
        """

        # Pad audio features
        input_features = [{"input_features": feat["input_features"]} for feat in features]
        batch = self.processor.feature_extractor.pad(
            input_features, padding=True, return_tensors="pt"
        )

        # Pad text labels
        labels = [{"input_ids": feat["labels"]} for feat in features]
        labels_batch = self.processor.tokenizer.pad(labels, padding=True, return_tensors="pt")

        # Replace padding token IDs with -100 so they are ignored in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Optionally remove BOS token if present at the beginning
        if (
            labels.size(1) > 1
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


from transformers import WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import prepare_model_for_kbit_training
import wandb

# Initialize the data collator to pad variable-length audio/text inputs
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=dataset.processor)

# Define training hyperparameters and settings
training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",  # Directory to save model checkpoints
    per_device_train_batch_size=8,  # Batch size per GPU
    gradient_accumulation_steps=1,  # Accumulate gradients for effective batch size
    learning_rate=1e-3,  # Learning rate
    warmup_steps=0,  # Number of warmup steps for learning rate scheduler
    num_train_epochs=3,  # Total number of training epochs
    logging_strategy="steps",  # Log every few steps
    logging_first_step=True,  # Log the very first training step
    logging_nan_inf_filter=False,  # Don’t filter NaN/inf in logs
    eval_steps=500,  # Run evaluation every 500 steps
    fp16=True,  # Use mixed-precision (FP16) training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    generation_max_length=128,  # Max length for generation during eval
    logging_steps=1,  # Log every step
    remove_unused_columns=False,  # Needed for PEFT since forward signature is modified
    label_names=["labels"],  # Tells Trainer to pass labels explicitly
)

# Load the Whisper model with 8-bit quantization and map to available devices (e.g., GPUs)
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny", load_in_8bit=True, device_map="auto"
)

# Prepare the model for LoRA-compatible 8-bit training (freezing norms, casting types)
model = prepare_model_for_kbit_training(model)

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
config = LoraConfig(
    r=32,  # Rank of LoRA decomposition
    lora_alpha=64,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention projections
    lora_dropout=0.05,  # Dropout applied to LoRA layers
    bias="none",  # Don't adapt bias terms
)

# Wrap the base model with LoRA using the above config
model = get_peft_model(model, config)
model.print_trainable_parameters()  # Print which parameters are trainable

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     return {"eval_loss": trainer.evaluate()["eval_loss"]}

# Initialize Hugging Face Trainer for training and evaluation
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    processing_class=dataset.processor.feature_extractor,  # Optional; may be unused
)

# Disable caching to avoid warnings during training (re-enable for inference)
model.config.use_cache = False

# Start training
trainer.train()
