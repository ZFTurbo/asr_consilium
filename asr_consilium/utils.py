import os
from tqdm import tqdm
import soundfile as sf
import json
import evaluate
from .normalizer import EnglishTextNormalizer, BasicMultilingualTextNormalizer


def get_dynamic_batches(
    items,
    batch_size=16
):
    total = len(items)
    i = 0
    while i < total:
        yield items[i: i + batch_size]
        i += batch_size


def get_dynamic_batches_advanced(
    items,
    available_mem=16
):
    """
    Get batches
    :param items:
    :param available_mem: your GPU VRAM size in GB
    :return:
    """

    MAX_BATCH_SIZE = 256
    total = len(items)
    i = 0

    while i < total:
        if items[i]['duration'] > 200:
            batch_size = 1
        else:
            batch_size = int(20 * available_mem / items[i]['duration']) + 1
            if batch_size > MAX_BATCH_SIZE:
                batch_size = MAX_BATCH_SIZE

        yield items[i: i + batch_size]
        i += batch_size


def store_results(results, output_file):
    out = open(output_file, "w", encoding='utf-8')
    for audio, text in results.items():
        r = {"audio": audio, "text": text}
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
    out.close()


def store_test_dataset_as_files(dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    output_jsonl_file = os.path.join(out_dir, "markdown.jsonl")
    if os.path.isfile(output_jsonl_file):
        print("Dataset already created!")
        return output_jsonl_file
    out = open(output_jsonl_file, 'w', encoding='utf-8')
    print("Dataset length: {}".format(len(dataset['test'])))
    for i in tqdm(range(len(dataset['test']))):
        orig_name = dataset['test'][i]["audio"]["path"]
        audio = dataset['test'][i]["audio"]["array"]
        sr = dataset['test'][i]["audio"]["sampling_rate"]
        sf.write(os.path.join(out_dir, orig_name), audio, sr, 'FLOAT')
        res = {
            'audio': orig_name,
            'text': dataset['test'][i]['text'],
            'duration': len(audio) / sr,
        }
        out.write(json.dumps(res, ensure_ascii=False) + '\n')
    out.close()
    return output_jsonl_file


def store_test_dataset_as_files_unique(dataset, out_dir, name='test'):
    os.makedirs(out_dir, exist_ok=True)
    output_jsonl_file = os.path.join(out_dir, "markdown.jsonl")
    if os.path.isfile(output_jsonl_file):
        print("Dataset already created!")
        return output_jsonl_file
    out = open(output_jsonl_file, 'w', encoding='utf-8')
    print(dataset)
    print("Dataset length: {}".format(len(dataset[name])))
    for i in tqdm(range(len(dataset[name]))):
        # print(dataset[name][i])
        if dataset[name][i]["audio"]["path"] is None:
            orig_name = '{}.wav'.format(i)
        else:
            part = dataset[name][i]["audio"]["path"][:-4]
            part = part.replace(":", "")
            orig_name = os.path.basename(part + '_{}.wav'.format(i))
        audio = dataset[name][i]["audio"]["array"]
        sr = dataset[name][i]["audio"]["sampling_rate"]
        # print(out_dir, orig_name, os.path.join(os.path.abspath(out_dir), orig_name))
        sf.write(os.path.join(out_dir, orig_name), audio, sr, 'FLOAT')
        res = {
            'audio': orig_name,
            'text': dataset[name][i]['text'],
            'duration': len(audio) / sr,
        }
        out.write(json.dumps(res, ensure_ascii=False) + '\n')
    out.close()
    return output_jsonl_file


def calc_metrics(
        target_file,
        preds_file,
        lang='en',
        verbose=True
):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    if lang == 'en':
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicMultilingualTextNormalizer()

    lines = open(target_file, 'r', encoding="utf-8").readlines()
    target = [json.loads(line) for line in lines]
    lines = open(preds_file, 'r', encoding="utf-8").readlines()
    preds = [json.loads(line) for line in lines]

    if verbose:
        print('Target entries: {} Prediction entries: {}'.format(len(target), len(preds)))

    target_ids = set()
    target_dict = {}
    for t in target:
        target_ids |= set([t['audio']])
        target_dict[t['audio']] = t['text']

    preds_ids = set()
    preds_dict = {}
    for t in preds:
        preds_ids |= set([t['audio']])
        preds_dict[t['audio']] = t['text']

    check = target_ids - preds_ids
    if len(check) != 0:
        print("Some problem here. Some ids wasn't predicted! {}".format(len(check)))
        print(list(check)[:5])
        print(target_file, preds_file)

    references1 = []
    hypotheses1 = []
    for audio_id in list(target_ids):
        references1.append(target_dict[audio_id])
        hypotheses1.append(preds_dict[audio_id])

    references = [normalizer(ref) for ref in references1]
    hypotheses = [normalizer(pred) for pred in hypotheses1]

    references_fixed = []
    hypotheses_fixed = []
    for r, p in zip(references, hypotheses):
        if r == '':
            references_fixed.append('a')
            if p == '':
                hypotheses_fixed.append('a')
            else:
                hypotheses_fixed.append(p)
        else:
            references_fixed.append(r)
            hypotheses_fixed.append(p)

    references = references_fixed
    hypotheses = hypotheses_fixed

    score_wer = wer_metric.compute(
        references=references,
        predictions=hypotheses
    )
    score_wer = round(100 * score_wer, 6)

    score_cer = cer_metric.compute(
        references=references,
        predictions=hypotheses
    )
    score_cer = round(100 * score_cer, 6)

    return score_wer, score_cer

