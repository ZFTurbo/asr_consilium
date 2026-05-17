import os

AVAILABLE_MODELS = [
    'nvidia/parakeet-tdt-0.6b-v2',
    'nvidia/parakeet-tdt-0.6b-v3',
    'nvidia/parakeet-tdt-1.1b',
    'Qwen/Qwen3-ASR-1.7B',
    'Qwen/Qwen3-ASR-0.6B',
    'nvidia/canary-qwen-2.5b',
    'ibm-granite/granite-speech-3.3-8b',
    'ibm-granite/granite-4.0-1b-speech',
    'ibm-granite/granite-speech-4.1-2b',
    'CohereLabs/cohere-transcribe-03-2026',
    'ZFTurbo/Phi-4-multimodal-instruct',
    'mistralai/Voxtral-Small-24B-2507',
    'mistralai/Voxtral-Mini-3B-2507',
]


def inference(
        jsonl_file,
        out_file,
        batch_size=16,
        model_list=None,
        weights=None,
        language=None,
        normalize=True,
        char_level=False,
        ensemble_type='median_extended',
        skip_existed=True,
):
    from .ensemble_functions import ensebmle_on_files
    from .utils import calc_metrics
    from .inference_canary import proc_data_with_canary
    try:
        from .inference_cohere import proc_data_with_cohere
    except Exception as e:
        print("Can't load cohere model. Exception: {}".format(e))
    from .inference_ibm_granite import proc_data_with_ibm_granite
    from .inference_parakeet import proc_data_with_parakeet
    from .inference_qwen3_asr import proc_data_with_qwen
    from .inference_phi4 import proc_data_with_microsoft_phi4
    from .inference_voxtral import proc_data_with_voxtral

    if model_list is None:
        if language is None or language == 'en':
            model_list = [
                'nvidia/parakeet-tdt-0.6b-v2',
                'nvidia/parakeet-tdt-0.6b-v3',
                'Qwen/Qwen3-ASR-1.7B',
                'nvidia/canary-qwen-2.5b',
                'ibm-granite/granite-speech-3.3-8b',
                'ibm-granite/granite-4.0-1b-speech',
                'ibm-granite/granite-speech-4.1-2b',
                'ZFTurbo/Phi-4-multimodal-instruct',
            ]
            weights = [4.5, 4.2, 8.4, 9.8, 8.7, 3.5, 8.9, 9.4]
            # Ensemble. WER: 4.6659 CER: 2.6183
        elif language == 'fr':
            model_list = [
                'nvidia/parakeet-tdt-0.6b-v3',
                'Qwen/Qwen3-ASR-1.7B',
                'ibm-granite/granite-speech-3.3-8b',
                'ibm-granite/granite-4.0-1b-speech',
                'ibm-granite/granite-speech-4.1-2b',
                'ZFTurbo/Phi-4-multimodal-instruct',
                'mistralai/Voxtral-Mini-3B-2507',
            ]
            weights = [7.8, 6.5, 3.9, 7.4, 6.7, 10.0, 5.0]
            # Ensemble. WER: 3.4755 CER: 1.4968
        elif language == 'de':
            model_list = [
                'nvidia/parakeet-tdt-0.6b-v3',
                'Qwen/Qwen3-ASR-1.7B',
                'ibm-granite/granite-speech-3.3-8b',
                'ibm-granite/granite-4.0-1b-speech',
                'ibm-granite/granite-speech-4.1-2b',
                'ZFTurbo/Phi-4-multimodal-instruct',
                'mistralai/Voxtral-Mini-3B-2507',
            ]
            weights = [6.2, 7.4, 1.8, 9.1, 7.5, 7.5, 6.5]
            # Ensemble. WER: 2.8811 CER: 1.0368
        elif language == 'it':
            model_list = [
                'nvidia/parakeet-tdt-0.6b-v3',
                'Qwen/Qwen3-ASR-1.7B',
                'ibm-granite/granite-speech-4.1-2b',
                'ZFTurbo/Phi-4-multimodal-instruct',
                'mistralai/Voxtral-Mini-3B-2507',
            ]
            weights = [4.7, 4.3, 9.4, 10.0, 6.4]
            # Ensemble.WER: 3.7433 CER: 1.1073
        elif language == 'es':
            model_list = [
                'nvidia/parakeet-tdt-0.6b-v3',
                'Qwen/Qwen3-ASR-1.7B',
                'ibm-granite/granite-speech-3.3-8b',
                'ibm-granite/granite-4.0-1b-speech',
                'ibm-granite/granite-speech-4.1-2b',
                'ZFTurbo/Phi-4-multimodal-instruct',
                'mistralai/Voxtral-Mini-3B-2507',
            ]
            weights = [8.3, 4.0, 7.0, 5.5, 8.3, 8.3, 9.5]
            # Ensemble.WER: 2.4235  CER: 0.9434
        elif language == 'pt':
            model_list = [
                'nvidia/parakeet-tdt-0.6b-v3',
                'Qwen/Qwen3-ASR-1.7B',
                'ibm-granite/granite-speech-3.3-8b',
                'ibm-granite/granite-4.0-1b-speech',
                'ibm-granite/granite-speech-4.1-2b',
                'ZFTurbo/Phi-4-multimodal-instruct',
                'mistralai/Voxtral-Mini-3B-2507',
            ]
            weights = [3.0, 3.2, 1.3, 1.4, 4.5, 4.3, 3.8]
            # Ensemble. WER: 2.7449 CER: 1.1316
        elif language == 'ru':
            model_list = [
                'nvidia/parakeet-tdt-0.6b-v3',
                'Qwen/Qwen3-ASR-1.7B',
                'mistralai/Voxtral-Mini-3B-2507',
            ]
            weights = [7.6, 7.5, 3.5]


    if weights is None:
        weights = [1.0] * len(model_list)

    print("Use following set of models for language:")
    for i, m in enumerate(model_list):
        print("Model: {} Weight: {}".format(m, weights[i]))

    files_for_ensemble = []
    for model in model_list:
        print("Go for model: {}".format(model))
        save_path = out_file[:-6] + '_' + model.split('/')[-1] + '.jsonl'
        files_for_ensemble.append(save_path)

        if skip_existed and os.path.isfile(save_path):
            print("File already exists: {}. Skip processing for {}".format(save_path, model))
            continue

        if 'parakeet-tdt' in model:
            proc_data_with_parakeet(
                jsonl_file,
                save_path,
                batch_size=batch_size,
                model_path=model,
            )
        elif 'Qwen3-ASR' in model:
            proc_data_with_qwen(
                jsonl_file,
                save_path,
                batch_size=batch_size,
                model_path=model,
                language=language,
            )
        elif 'canary-qwen' in model:
            proc_data_with_canary(
                jsonl_file,
                save_path,
                batch_size=batch_size,
                model_path=model,
            )
        elif 'granite-' in model:
            proc_data_with_ibm_granite(
                jsonl_file,
                save_path,
                batch_size=batch_size,
                model_path=model,
            )
        elif 'cohere-transcribe' in model:
            proc_data_with_cohere(
                jsonl_file,
                save_path,
                batch_size=batch_size,
                model_path=model,
            )
        elif 'Phi-4' in model:
            proc_data_with_microsoft_phi4(
                jsonl_file,
                save_path,
                batch_size=batch_size,
                model_path=model,
            )
        elif 'Voxtral' in model:
            proc_data_with_voxtral(
                jsonl_file,
                save_path,
                batch_size=batch_size,
                model_path=model,
            )

        try:
            # If you have "text" field in input jsonl then you can calc metrics
            score_wer, score_cer = calc_metrics(jsonl_file, save_path)
            print("Model: {} WER: {:.4f} CER: {:.4f}".format(model, score_wer, score_cer))
        except Exception as e:
            print("Metrics are not available... Error:", str(e))

    ensebmle_on_files(
        files_for_ensemble,
        out_file,
        normalize=normalize,
        char_level=char_level,
        weights=weights,
        language=language,
        ensemble_type=ensemble_type
    )

    try:
        # If you have "text" field in input jsonl then you can calc metrics
        score_wer, score_cer = calc_metrics(jsonl_file, out_file)
        print("Ensemble. WER: {:.4f} CER: {:.4f}".format(score_wer, score_cer))
    except Exception as e:
        print("Metrics are not available... Error:", str(e))
