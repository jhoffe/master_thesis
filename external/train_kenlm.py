# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# This script would train an N-gram language model with KenLM library (https://github.com/kpu/kenlm) which can be used
# with the beam search decoders on top of the ASR models. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# After the N-gram model is trained, and stored in the binary format, you may use
# 'scripts/ngram_lm/eval_beamsearch_ngram.py' to evaluate it on an ASR model.
#
# You need to install the KenLM library and also the beam search decoders to use this feature. Please refer
# to 'scripts/ngram_lm/install_beamsearch_decoders.sh' on how to install them.
#
# USAGE: python train_kenlm.py nemo_model_file=<path to the .nemo file of the model> \
#                              train_paths=<list of paths to the training text or JSON manifest file> \
#                              kenlm_bin_path=<path to the bin folder of KenLM library> \
#                              kenlm_model_file=<path to store the binary KenLM model> \
#                              ngram_length=<order of N-gram model> \
#
# After training is done, the binary LM model is stored at the path specified by '--kenlm_model_file'.
# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

from dataclasses import dataclass, field
from glob import glob
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import List

from joblib import Parallel, delayed
from loguru import logger
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel, kenlm_utils
from nemo.collections.asr.parts.submodules.ngram_lm.constants import DEFAULT_TOKEN_OFFSET
from nemo.collections.common.tokenizers import AggregateTokenizer
from nemo.core.config import hydra_runner
import numpy as np
from omegaconf import MISSING
import requests
import torch
from tqdm.auto import tqdm

"""
NeMo's beam search decoders only support char-level encodings. In order to make it work with BPE-level encodings, we
use a trick to encode the sub-word tokens of the training data as unicode characters and train a char-level KenLM. 
"""

# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Utility methods to be used for training N-gram LM with KenLM in train_kenlm.py

The BPE sub-words are encoded using the Unicode table. 
This encoding scheme reduces the required memory significantly, and the LM and its binary blob format require less storage space. 
The value DEFAULT_TOKEN_OFFSET from nemo.collections.asr.parts.submodules.ctc_beam_decoding is utilized as the offset value.
"""

CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512

# List of the supported models to be used with N-gram LM and beam search decoding
SUPPORTED_MODELS = {
    "EncDecCTCModelBPE": "subword",
    "EncDecCTCModel": "char",
    "EncDecRNNTBPEModel": "subword",
    "EncDecRNNTModel": "char",
    "EncDecHybridRNNTCTCBPEModel": "subword",
    "EncDecHybridRNNTCTCModel": "char",
    "EncDecMultiTaskModel": "subword",
}


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])


def get_train_list(args_train_path):
    train_path = []
    for train_item in args_train_path:
        if os.path.isdir(train_item):
            file_list = os.listdir(train_item)
            train_path.extend([os.path.join(train_item, file) for file in file_list])

        elif os.path.isfile(train_item):
            train_path.append(train_item)
    return sorted(train_path)


def setup_tokenizer(nemo_model_file):
    """TOKENIZER SETUP
    nemo_model_file (str): The path to the NeMo model file (.nemo).
    """
    logger.info(f"Loading nemo model '{nemo_model_file}' ...")
    if nemo_model_file.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(
            nemo_model_file, map_location=torch.device("cpu")
        )
    else:
        logger.warning(
            "tokenizer_model_file does not end with .model or .nemo, therefore trying to load a pretrained model with this name."
        )
        model = nemo_asr.models.ASRModel.from_pretrained(
            nemo_model_file, map_location=torch.device("cpu")
        )

    is_aggregate_tokenizer = False
    tokenizer_nemo = None
    full_vocab_size = None
    encoding_level = SUPPORTED_MODELS.get(type(model).__name__, None)
    if not encoding_level:
        logger.warning(
            f"Model type '{type(model).__name__}' may not be supported. Would try to train a char-level LM."
        )
        encoding_level = "char"

    if encoding_level == "subword":
        is_aggregate_tokenizer = isinstance(model.tokenizer, AggregateTokenizer)
        tokenizer_nemo = model.tokenizer

        full_vocab_size = tokenizer_nemo.vocab_size

        # sanity check for LM (blank_id == vocab_size)
        if isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
            assert full_vocab_size == model.decoding.decoding.blank_id
        elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
            assert full_vocab_size == model.decoding.decoding._blank_index
        elif isinstance(model, nemo_asr.models.EncDecHybridRNNTCTCBPEModel):
            try:
                # rnnt head
                assert full_vocab_size == model.decoding.decoding._blank_index
            except AttributeError:
                # ctc head
                assert full_vocab_size == model.decoding.decoding.blank_id
        elif isinstance(model, nemo_asr.models.EncDecMultiTaskModel):
            assert full_vocab_size == model.decoding.decoding.beam_search.num_tokens
        else:
            logger.warning(f"Unknown type of model {type(model).__name__}")

    del model

    return tokenizer_nemo, encoding_level, is_aggregate_tokenizer, full_vocab_size


def iter_files(source_path, dest_path, tokenizer, encoding_level, is_aggregate_tokenizer, verbose):
    if isinstance(dest_path, list):
        paths = zip(dest_path, source_path)
    else:  # dest_path is stdin of KenLM
        paths = [(dest_path, path) for path in source_path]

    for dest_path, input_path in paths:
        dataset = read_train_file(
            input_path, is_aggregate_tokenizer=is_aggregate_tokenizer, verbose=verbose
        )
        if encoding_level == "subword":
            tokenize_text(
                data=dataset,
                tokenizer=tokenizer,
                path=dest_path,
                chunk_size=CHUNK_SIZE,
                buffer_size=CHUNK_BUFFER_SIZE,
            )
        else:  # encoding_level == "char"
            if isinstance(dest_path, str):
                with open(dest_path, "w", encoding="utf-8") as f:
                    for line in dataset:
                        f.write(line[0] + "\n")
            else:  # write to stdin of KenLM
                for line in dataset:
                    dest_path.write((line[0] + "\n").encode())


def read_train_file(
    path,
    is_aggregate_tokenizer: bool = False,
    verbose: int = 1,
):
    lines_read = 0
    text_dataset, lang_dataset = [], []
    if path[-8:] == ".json.gz":  # for Common Crawl dataset
        fin = gzip.open(path, "r")
    else:
        fin = open(path, "r", encoding="utf-8")

    if verbose > 0:
        reader = tqdm(iter(lambda: fin.readline(), ""), desc="Read 0 lines", unit=" lines")
    else:
        reader = fin

    for line in reader:
        lang = None
        if line:
            if path[-8:] == ".json.gz":  # for Common Crawl dataset
                line = json.loads(line.decode("utf-8"))["text"]
            elif path.endswith(".json") or path.endswith(".jsonl"):
                jline = json.loads(line)
                line = jline["text"]
                if is_aggregate_tokenizer:
                    lang = jline["lang"]

            line_list = line.split("\n")

            line = " ".join(line_list)
            if line:
                text_dataset.append(line)
                if lang:
                    lang_dataset.append(lang)
                lines_read += 1
                if verbose > 0 and lines_read % 100000 == 0:
                    reader.set_description(f"Read {lines_read} lines")
        else:
            break
    fin.close()
    if is_aggregate_tokenizer:
        assert len(text_dataset) == len(lang_dataset), (
            f"text_dataset length {len(text_dataset)} and lang_dataset length {len(lang_dataset)} must be the same!"
        )
        return list(zip(text_dataset, lang_dataset))
    else:
        return [[text] for text in text_dataset]


def tokenize_str(texts, tokenizer):
    tokenized_text = []
    for text in texts:
        tok_text = tokenizer.text_to_ids(*text)
        tok_text = [chr(token + DEFAULT_TOKEN_OFFSET) for token in tok_text]
        tokenized_text.append(tok_text)
    return tokenized_text


def tokenize_text(data, tokenizer, path, chunk_size=8192, buffer_size=32):
    dataset_len = len(data)
    current_step = 0
    if isinstance(path, str) and os.path.exists(path):
        os.remove(path)

    with Parallel(n_jobs=-2, verbose=0) as parallel:
        while True:
            start = current_step * chunk_size
            end = min((current_step + buffer_size) * chunk_size, dataset_len)

            tokenized_data = parallel(
                delayed(tokenize_str)(data[start : start + chunk_size], tokenizer)
                for start in range(start, end, chunk_size)
            )

            # Write dataset
            write_dataset(tokenized_data, path)
            current_step += len(tokenized_data)
            logger.info(
                f"Finished writing {len(tokenized_data)} chunks to {path}. Current chunk index = {current_step}"
            )
            del tokenized_data
            if end >= dataset_len:
                break


def write_dataset(chunks, path):
    if isinstance(path, str):
        with open(path, "a+", encoding="utf-8") as f:
            for chunk_idx in tqdm(
                range(len(chunks)), desc="Chunk ", total=len(chunks), unit=" chunks"
            ):
                for text in chunks[chunk_idx]:
                    line = " ".join(text)
                    f.write(f"{line}\n")
    else:  # write to stdin of KenLM
        for chunk_idx in range(len(chunks)):
            for text in chunks[chunk_idx]:
                line = " ".join(text)
                path.write((line + "\n").encode())


def download_and_compile_kenlm() -> Path:
    """Download and compile the `kenlm` library.

    Args:
        config:
            Hydra configuration dictionary.

    Returns:
        Path to the `kenlm` build directory.
    """
    # Ensure that the `kenlm` directory exists, and download if otherwise
    cache_dir = Path.home() / ".cache"

    kenlm_dir = cache_dir / "kenlm"
    if not kenlm_dir.exists():
        logger.info("Downloading `kenlm`...")
        with requests.get(url="https://kheafield.com/code/kenlm.tar.gz") as response:
            response.raise_for_status()
            data = response.content
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(path=cache_dir)

    # Compile `kenlm` if it hasn't already been compiled
    kenlm_build_dir = kenlm_dir / "build"
    if not (kenlm_build_dir / "bin" / "lmplz").exists():
        logger.info("Compiling `kenlm`...")
        kenlm_build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=str(kenlm_build_dir))
        subprocess.run(["make", "-j", "4"], cwd=str(kenlm_build_dir))

    logger.info(f"`kenlm` is ready at {kenlm_build_dir}/bin")

    return kenlm_build_dir / "bin"


@dataclass
class TrainKenlmConfig:
    """
    Train an N-gram language model with KenLM to be used with beam search decoder of ASR models.
    """

    train_paths: List[str] = (
        MISSING  # List of training files or folders. Files can be a plain text file or ".json" manifest or ".json.gz". Example: [/path/to/manifest/file,/path/to/folder]
    )

    nemo_model_file: str = (
        MISSING  # The path to '.nemo' file of the ASR model, or name of a pretrained NeMo model
    )
    kenlm_model_file: str = MISSING  # The path to store the KenLM binary model file
    ngram_length: int = MISSING  # The order of N-gram LM

    preserve_arpa: bool = False  # Whether to preserve the intermediate ARPA file.
    ngram_prune: List[int] = field(
        default_factory=lambda: [0]
    )  # List of digits to prune Ngram. Example: [0,0,1]. See Pruning section on the https://kheafield.com/code/kenlm/estimation
    cache_path: str = ""  # Cache path to save tokenized files.
    verbose: int = 1  # Verbose level, default is 1.
    save_nemo: bool = False  # Save .nemo checkpoint to use with NGramGPULanguageModel
    normalize_unk_nemo: bool = True  # Normalize the UNK token in the NGramGPULanguageModel model


@hydra_runner(config_path=None, config_name="TrainKenlmConfig", schema=TrainKenlmConfig)
def main(args: TrainKenlmConfig):
    train_paths = get_train_list(args.train_paths)
    print(train_paths, args.train_paths)

    kenlm_bin_path = download_and_compile_kenlm()

    if isinstance(args.ngram_prune, str):
        args.ngram_prune = [args.ngram_prune]

    tokenizer, encoding_level, is_aggregate_tokenizer, full_vocab_size = setup_tokenizer(
        args.nemo_model_file
    )

    if encoding_level == "subword":
        discount_arg = "--discount_fallback"  # --discount_fallback is needed for training KenLM for BPE-based models
    else:
        discount_arg = ""

    arpa_file = f"{args.kenlm_model_file}.tmp.arpa"
    """ LMPLZ ARGUMENT SETUP """
    kenlm_args = [
        os.path.join(kenlm_bin_path, "lmplz"),
        "-o",
        str(args.ngram_length),
        "--arpa",
        arpa_file,
        discount_arg,
        "--prune",
    ] + [str(n) for n in args.ngram_prune]

    if args.cache_path:
        if not os.path.exists(args.cache_path):
            os.makedirs(args.cache_path, exist_ok=True)

        """ DATASET SETUP """
        encoded_train_files = []
        for file_num, train_file in enumerate(train_paths):
            logger.info(
                f"Encoding the train file '{train_file}' number {file_num + 1} out of {len(train_paths)} ..."
            )

            cached_files = glob(os.path.join(args.cache_path, os.path.split(train_file)[1]) + "*")
            encoded_train_file = os.path.join(
                args.cache_path, os.path.split(train_file)[1] + f"_{file_num}.tmp.txt"
            )
            if (
                cached_files and cached_files[0] != encoded_train_file
            ):  # cached_files exists but has another file name: f"_{file_num}.tmp.txt"
                os.rename(cached_files[0], encoded_train_file)
                logger.info("Rename", cached_files[0], "to", encoded_train_file)

            encoded_train_files.append(encoded_train_file)

        kenlm_utils.iter_files(
            source_path=train_paths,
            dest_path=encoded_train_files,
            tokenizer=tokenizer,
            encoding_level=encoding_level,
            is_aggregate_tokenizer=is_aggregate_tokenizer,
            verbose=args.verbose,
        )

        first_process_args = ["cat"] + encoded_train_files
        first_process = subprocess.Popen(
            first_process_args, stdout=subprocess.PIPE, stderr=sys.stderr
        )

        logger.info(f"Running lmplz command \n\n{' '.join(kenlm_args)}\n\n")
        kenlm_p = subprocess.run(
            kenlm_args,
            stdin=first_process.stdout,
            capture_output=False,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        first_process.wait()

    else:
        logger.info(f"Running lmplz command \n\n{' '.join(kenlm_args)}\n\n")
        kenlm_p = subprocess.Popen(
            kenlm_args, stdout=sys.stdout, stdin=subprocess.PIPE, stderr=sys.stderr
        )
        logger.info(f"Is aggregate tokenizer: {is_aggregate_tokenizer}")

        iter_files(
            source_path=train_paths,
            dest_path=kenlm_p.stdin,
            tokenizer=tokenizer,
            encoding_level=encoding_level,
            is_aggregate_tokenizer=is_aggregate_tokenizer,
            verbose=args.verbose,
        )

        kenlm_p.communicate()

    if kenlm_p.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")

    """ BINARY BUILD """

    kenlm_args = [
        os.path.join(kenlm_bin_path, "build_binary"),
        "trie",
        arpa_file,
        args.kenlm_model_file,
    ]
    logger.info(f"Running binary_build command \n\n{' '.join(kenlm_args)}\n\n")
    ret = subprocess.run(
        kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr
    )

    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")

    if args.save_nemo:
        if full_vocab_size is None:
            raise NotImplementedError("Unknown vocab size, cannot convert the model")
        nemo_model = NGramGPULanguageModel.from_arpa(
            lm_path=arpa_file, vocab_size=full_vocab_size, normalize_unk=args.normalize_unk_nemo
        )
        nemo_model.save_to(f"{args.kenlm_model_file}.nemo")

    if not args.preserve_arpa:
        os.remove(arpa_file)
        logger.info(f"Deleted the arpa file '{arpa_file}'.")


if __name__ == "__main__":
    main()
