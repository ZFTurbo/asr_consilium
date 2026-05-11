import os
import torch
import time
from tqdm import tqdm
import json
import logging
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("nemo").setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)
from nemo.utils import logging as nemo_logging
nemo_logging.setLevel("ERROR")
import nemo.collections.asr as nemo_asr
from .utils import get_dynamic_batches, store_results


class ParakeetModel:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_path):
        print(f"Loading model from: {model_path}")
        # model = nemo_asr.models.ASRModel.restore_from(model_path)
        model = nemo_asr.models.ASRModel.from_pretrained(model_path)
        return cls(model)

    def predict(self, audio_path):
        hypotheses = self.model.transcribe([str(audio_path)], verbose=False)
        pred = hypotheses[0].text
        return pred

    def predict_batch(self, data: list[str], abs_path, batch_size: int = 4):
        d = [abs_path + '/' + str(p['audio']) for p in data]
        hypotheses = self.model.transcribe(
            d,
            batch_size=batch_size,
            verbose=False,
        )
        preds = [hyp.text for hyp in hypotheses]
        return preds


def proc_data_with_parakeet(
        jsonl_file,
        out_file,
        batch_size=16,
        model_path="nvidia/parakeet-tdt-0.6b-v2",
):
    """
    :param data: list of dicts with {'audio_path': ..., 'duration': ...}
    :param cache_dir:
    :param model_path: parakeet model path. Choose from:
        nvidia/parakeet-tdt-0.6b-v2 (Default)
        nvidia/parakeet-tdt-0.6b-v3
        nvidia/parakeet-tdt-1.1b
    :return:
    """
    model = ParakeetModel.load(model_path)

    lines = open(jsonl_file, 'r', encoding="utf-8").readlines()
    items = [json.loads(line) for line in lines]
    items.sort(key=lambda x: x["duration"], reverse=True)

    start_time_overall = time.time()
    # Predict
    predictions = {}
    processed = 0
    abs_path = os.path.dirname(jsonl_file)
    with torch.inference_mode():
        with tqdm(total=len(items)) as pbar:
            for batch in get_dynamic_batches(items, batch_size):
                preds = model.predict_batch(
                    batch,
                    abs_path,
                    batch_size=len(batch),
                )
                for item, pred in zip(batch, preds):
                    predictions[item["audio"]] = pred
                this_batch_size = len(batch)
                pbar.update(this_batch_size)
                processed += this_batch_size

    print("Transcription complete in {:.2f} seconds".format(time.time() - start_time_overall))
    if out_file is not None:
        store_results(
            predictions,
            out_file
        )
    return predictions

