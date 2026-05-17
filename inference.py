import os

if __name__ == '__main__':
    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    code_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.environ['HF_HOME'] = code_path + 'local_models_cache/'


from asr_consilium import inference, AVAILABLE_MODELS
import argparse
import sys
import torch


if __name__ == "__main__":
    # Diagnostics
    print("Torch version: {}".format(torch.__version__))
    print("CUDA available: {}".format(torch.cuda.is_available()))
    print("CUDA device count: {}".format(torch.cuda.device_count()))

    parser = argparse.ArgumentParser(description="Ensemble ASR models script")

    # Paths and core parameters
    parser.add_argument("--input_data", required=True, type=str, help="Path to markdown.jsonl")
    parser.add_argument("--output", required=True, type=str, help="Path to output file where results will be stored")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")

    # Model selection and ensembling configuration
    parser.add_argument('--models', nargs='+', help='Model names for ensembling')
    parser.add_argument('--list', action='store_true', help='Show list of available models and exit')
    parser.add_argument('--weights', nargs='+', type=float, help='Weights for models (must match the number of models)')

    # Metadata and processing settings
    parser.add_argument('--language', default='en', help='Language for models (default: en). Supported: en, fr, de, it, es, pt, ru, xx (other)')

    # Normalization flags (Defaults to True)
    # Using --no-normalize will set normalize to False
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable text normalization before ensembling")
    parser.set_defaults(normalize=True)

    # Granularity level (Defaults to False/Word-level)
    parser.add_argument("--char_level", action="store_true", help="Use character level for ensembling instead of word level")

    # Checkpointing (Defaults to True)
    # Using --force-recalculate will set skip_existed to False
    parser.add_argument("--force-recalculate", dest="skip_existed", action="store_false", help="Recalculate every model even if results already exist")
    parser.set_defaults(skip_existed=True)
    args = parser.parse_args()

    if args.list:
        print("List of available models:")
        for model in AVAILABLE_MODELS:
            print(model)
        sys.exit(0)

    if args.models and args.weights and len(args.models) != len(args.weights):
        parser.error("The number of weights must match the number of models!")
        sys.exit(0)

    print("Inference arguments: ", str(vars(args)))

    inference(
        jsonl_file=args.input_data,
        out_file=args.output,
        batch_size=args.batch_size,
        model_list=args.models,
        weights=args.weights,
        language=args.language,
        normalize=args.normalize,
        char_level=args.char_level,
        skip_existed=args.skip_existed,
    )
