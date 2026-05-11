import os
import torch
import time
from tqdm import tqdm
import json
import librosa
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from .utils import get_dynamic_batches, store_results


TOKENS_PER_SECOND = 20


def proc_data_with_microsoft_phi4(
        jsonl_file,
        out_file,
        batch_size=16,
        model_path="ZFTurbo/Phi-4-multimodal-instruct",
):
    """
    :param data: list of dicts with {'audio_path': ..., 'duration': ...}
    :param cache_dir:
    :param model_path: ibm granite model path. Choose from:
        microsoft/Phi-4-multimodal-instruct
        ZFTurbo/Phi-4-multimodal-instruct - fork with transformers==4.57.6 support
    :return:
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model: {}".format(model_path))

    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Use AutoModel to let the config determine the class
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        # _attn_implementation='flash_attention_2',
        _attn_implementation='sdpa',
    ).to(device)

    generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

    lines = open(jsonl_file, 'r', encoding="utf-8").readlines()
    items = [json.loads(line) for line in lines]
    items.sort(key=lambda x: x["duration"], reverse=True)

    start_time_overall = time.time()
    cur_time = time.time()

    speech_prompt = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
    USER_PROMPT_CONTENT = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'

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
                    batch_audios.append((audio, 16000))

                    # Content must be string
                    chat_batch.append(USER_PROMPT_CONTENT)

                    max_new_tokens = max(max_new_tokens, int(item["duration"] * TOKENS_PER_SECOND))
                    total += 1

                inputs = processor(
                    text=chat_batch,
                    audios=batch_audios,
                    return_tensors='pt'
                ).to(device)
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
                    generation_config=generation_config,
                    num_logits_to_keep=0,
                )

                input_len = inputs.input_ids.shape[1]
                generated_ids = output_ids[:, input_len:]
                decoded_preds = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                for item, pred in zip(batch, decoded_preds):
                    # print(pred)
                    predictions[item['audio']] = pred

    print("Transcription complete in {:.2f} seconds".format(time.time() - start_time_overall))
    if out_file is not None:
        store_results(
            predictions,
            out_file
        )
    return predictions

