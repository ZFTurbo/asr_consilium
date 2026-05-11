import os
import torch
import time
from tqdm import tqdm
import json
import librosa
from transformers import AutoModel, AutoProcessor, AutoModelForSpeechSeq2Seq
from .utils import get_dynamic_batches, store_results


TOKENS_PER_SECOND = 20


def proc_data_with_ibm_granite(
        jsonl_file,
        out_file,
        batch_size=16,
        model_path="ibm-granite/granite-4.0-1b-speech",
):
    """
    :param data: list of dicts with {'audio_path': ..., 'duration': ...}
    :param cache_dir:
    :param model_path: ibm granite model path. Choose from:
        ibm-granite/granite-4.0-1b-speech
        ibm-granite/granite-speech-3.3-8b
        ibm-granite/granite-speech-4.1-2b
    :return:
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model: {}".format(model_path))

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Use AutoModel to let the config determine the class
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
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

                text_prompts = [
                    processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                    for c in chat_batch
                ]

                inputs = processor(
                    text_prompts,
                    batch_audios,
                    # sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).to(model.device)

                pbar.update(len(batch))
                pbar.set_postfix({
                    "batch size": len(batch),
                    "duration (sec)": batch[0]["duration"],
                    "max tokens": max_new_tokens,
                    "iter time (sec)": time.time() - cur_time,
                })
                cur_time = time.time()

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )

                input_len = inputs.input_ids.shape[1]
                generated_ids = output_ids[:, input_len:]
                decoded_preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

                for item, pred in zip(batch, decoded_preds):
                    predictions[item['audio']] = pred

    print("Transcription complete in {:.2f} seconds".format(time.time() - start_time_overall))
    if out_file is not None:
        store_results(
            predictions,
            out_file
        )
    return predictions

