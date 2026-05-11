import os
import torch
import time
from tqdm import tqdm
import json
import librosa
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from .utils import get_dynamic_batches, store_results


TOKENS_PER_SECOND = 20


def proc_data_with_cohere(
        jsonl_file,
        out_file,
        batch_size=16,
        model_path="CohereLabs/cohere-transcribe-03-2026",
):
    """
    :param data: list of dicts with {'audio_path': ..., 'duration': ...}
    :param cache_dir:
    :param model_path: cohere model path. Choose from:
        CohereLabs/cohere-transcribe-03-2026
    :return:
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model: {}".format(model_path))

    processor = AutoProcessor.from_pretrained(
        model_path,
        token="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",
    )
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_path,
        token="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",
        device_map="auto",
    )

    lines = open(jsonl_file, 'r', encoding="utf-8").readlines()
    items = [json.loads(line) for line in lines]
    items.sort(key=lambda x: x["duration"], reverse=True)

    start_time_overall = time.time()
    cur_time = time.time()

    USER_PROMPT_CONTENT = "<|audio|>can you transcribe the speech into a written format?"

    # Predict
    predictions = {}
    total = 0
    abs_path = os.path.dirname(jsonl_file)
    with torch.inference_mode():
        with tqdm(total=len(items)) as pbar:
            for batch in get_dynamic_batches(items, batch_size):
                chat_batch = []
                batch_audios = []
                max_new_tokens = 0

                for item in batch:
                    path = str(abs_path + "/" + item["audio"])
                    audio, _ = librosa.load(path, sr=16000, mono=True)
                    batch_audios.append(audio)

                    # Content must be string
                    chat = [{"role": "user", "content": USER_PROMPT_CONTENT}]
                    chat_batch.append(chat)

                    max_new_tokens = max(max_new_tokens, int(item["duration"] * TOKENS_PER_SECOND))
                    total += 1

                inputs = processor(
                    batch_audios,
                    sampling_rate=16000,
                    return_tensors="pt",
                    language="en",
                    padding=True
                )
                inputs.to(model.device, dtype=model.dtype)

                pbar.update(len(batch))
                pbar.set_postfix({
                    "batch size": len(batch),
                    "duration (sec)": batch[0]["duration"],
                    "max tokens": max_new_tokens,
                    "iter time (sec)": time.time() - cur_time,
                })
                cur_time = time.time()

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )

                decoded_preds = processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )

                for item, pred in zip(batch, decoded_preds):
                    predictions[item['audio']] = pred

    print("Transcription complete in {:.2f} seconds".format(time.time() - start_time_overall))
    if out_file is not None:
        store_results(
            predictions,
            out_file
        )
    return predictions

