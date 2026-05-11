import os
import torch
import time
from tqdm import tqdm
import json
from nemo.collections.speechlm2.models import SALM
from .utils import get_dynamic_batches, store_results


TOKENS_PER_SECOND = 20


def proc_data_with_canary(
        jsonl_file,
        out_file=None,
        batch_size=16,
        model_path="nvidia/canary-qwen-2.5b",
):
    """
    :param data: list of dicts with {'audio_path': ..., 'duration': ...}
    :param cache_dir:
    :param model_path: canary model path. Choose from:
        nvidia/canary-qwen-2.5b
    :return:
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model: {}".format(model_path))
    model = SALM.from_pretrained(
        model_path,
    )
    model.to(device)
    model.bfloat16()
    model.eval()

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
                prompts = []
                max_new_tokens = 0
                for item in batch:
                    path = str(abs_path  + '/' + item["audio"])
                    prompts.append([{
                        "role": "user",
                        "content": f"Transcribe the following in English: {model.audio_locator_tag}",
                        "audio": [path]
                    }])
                    max_new_tokens = max(max_new_tokens, int(item["duration"] * TOKENS_PER_SECOND))
                    total += 1

                pbar.update(len(batch))
                pbar.set_postfix({
                    "batch size": len(batch),
                    "duration (sec)": batch[0]["duration"],
                    "max tokens": max_new_tokens,
                    "iter time (sec)": time.time() - cur_time,
                })
                cur_time = time.time()

                answer_ids = model.generate(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                )

                for item, ids in zip(batch, answer_ids):
                    pred = model.tokenizer.ids_to_text(ids.cpu())
                    predictions[item["audio"]] = pred
                    total += 1

    print("Transcription complete in {:.2f} seconds".format(time.time() - start_time_overall))
    if out_file is not None:
        store_results(
            predictions,
            out_file
        )
    return predictions

