import os
import torch
import time
from tqdm import tqdm
import json
from asr_consilium.qwen_asr import Qwen3ASRModel
from asr_consilium.qwen_asr.inference.utils import SUPPORTED_LANGUAGES
from .utils import get_dynamic_batches, store_results


TOKENS_PER_SECOND = 20

ISO_TO_LANGUAGE_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian"
}


def proc_data_with_qwen(
        jsonl_file,
        out_file,
        language="en",
        batch_size=16,
        model_path="Qwen/Qwen3-ASR-1.7B",
):
    """
    :param data: list of dicts with {'audio_path': ..., 'duration': ...}
    :param cache_dir:
    :param model_path: qwen model path. Choose from:
        Qwen/Qwen3-ASR-1.7B
        Qwen/Qwen3-ASR-0.6B
    :return:
    """

    # Convert language to Qwen3-ASR format
    if language in ISO_TO_LANGUAGE_MAP:
        language = ISO_TO_LANGUAGE_MAP[language]
    else:
        language = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model: {}".format(model_path))
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        # cache_dir=code_path + 'Qwen3-ASR-1.7B/',
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=128,
        max_new_tokens=-1,
    )

    lines = open(jsonl_file, 'r', encoding="utf-8").readlines()
    items = [json.loads(line) for line in lines]
    items.sort(key=lambda x: x["duration"], reverse=True)

    start_time_overall = time.time()
    cur_time = time.time()

    # Predict
    predictions = {}
    total = 0
    abs_path = os.path.dirname(jsonl_file)
    with torch.inference_mode():
        with tqdm(total=len(items)) as pbar:
            for batch in get_dynamic_batches(items, batch_size):
                paths = []
                languages = []
                max_new_tokens = 0
                for item in batch:
                    paths.append(abs_path + '/' + item['audio'])
                    languages.append(language)
                    # languages.append(None)
                    max_new_tokens = max(max_new_tokens, int(item["duration"] * TOKENS_PER_SECOND))

                pbar.update(len(batch))
                pbar.set_postfix({
                    "batch size": len(batch),
                    "duration (sec)": batch[0]["duration"],
                    "max tokens": max_new_tokens,
                    "iter time (sec)": time.time() - cur_time,
                })
                cur_time = time.time()

                results = model.transcribe(
                    audio=paths,
                    language=languages,  # can also be set to None for automatic language detection
                    return_time_stamps=False,
                    max_new_tokens=max_new_tokens,
                )

                for i, r in enumerate(results):
                    item = batch[i]
                    predictions[item["audio"]] = r.text
                    total += 1

    print("Transcription complete in {:.2f} seconds".format(time.time() - start_time_overall))
    if out_file is not None:
        store_results(
            predictions,
            out_file
        )
    return predictions

