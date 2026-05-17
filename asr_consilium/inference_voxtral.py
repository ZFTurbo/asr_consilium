import os
import torch
import time
from tqdm import tqdm
import json
import librosa
import numpy as np
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from .utils import get_dynamic_batches, store_results


TOKENS_PER_SECOND = 20


def proc_data_with_voxtral(
        jsonl_file,
        out_file,
        batch_size=16,
        language='en',
        model_path="mistralai/Voxtral-Mini-3B-2507",
):
    """
    :param data: list of dicts with {'audio_path': ..., 'duration': ...}
    :param cache_dir:
    :param model_path: voxtral model path. Choose from:
        mistralai/Voxtral-Small-24B-2507
        mistralai/Voxtral-Mini-3B-2507
    :return:
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model: {} Language: {}".format(model_path, language))

    processor = AutoProcessor.from_pretrained(model_path)
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
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
                max_new_tokens = 0
                all_inputs = []
                for item in batch:
                    path = str(abs_path + "/" + item["audio"])
                    input = processor.apply_transcription_request(
                        language=language,
                        audio=path,
                        model_id=model_path
                    )
                    all_inputs.append(input)
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

                input_ids_list = [inp.input_ids.squeeze(0) for inp in all_inputs]
                attention_mask_list = [inp.attention_mask.squeeze(0) for inp in all_inputs]

                max_len = max(ids.shape[0] for ids in input_ids_list)
                pad_id = processor.tokenizer.pad_token_id

                def left_pad(tensor, max_len, pad_value):
                    pad_size = max_len - tensor.shape[0]
                    if pad_size == 0:
                        return tensor
                    pad = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
                    return torch.cat([pad, tensor], dim=0)

                padded_input_ids = torch.stack(
                    [left_pad(ids, max_len, pad_id) for ids in input_ids_list]
                )
                padded_attention_mask = torch.stack(
                    [left_pad(mask, max_len, 0) for mask in attention_mask_list]
                )

                batch_inputs = {
                    "input_ids": padded_input_ids.to(device),
                    "attention_mask": padded_attention_mask.to(device),
                }

                extra_keys = [k for k in all_inputs[0].keys() if k not in ("input_ids", "attention_mask")]
                for key in extra_keys:
                    tensors = [inp[key] for inp in all_inputs]
                    try:
                        batch_inputs[key] = torch.cat(tensors, dim=0).to(device, dtype=torch.bfloat16)
                    except RuntimeError:
                        max_feat_len = max(t.shape[-1] for t in tensors)
                        padded = torch.stack([
                            torch.nn.functional.pad(t.squeeze(0), (0, max_feat_len - t.shape[-1]))
                            for t in tensors
                        ])
                        batch_inputs[key] = padded.to(device, dtype=torch.bfloat16)

                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                )

                decoded_preds = processor.batch_decode(
                    outputs[:, batch_inputs["input_ids"].shape[1]:],
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

