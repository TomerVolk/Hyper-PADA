import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from modeling import LoggingCallback, MODEL_CLASSES, MODEL_PREFIXES

import os
from utils.utils import set_seed, get_last_version, get_source
from box import Box
from argparse import ArgumentParser


OVERRIDE_GENERATOR = False


def define_parser():
    parser = ArgumentParser()
    parser.add_argument('--model_type', choices=list(MODEL_CLASSES.keys()), type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--task_name', type=str, choices=['sentiment', 'mnli', 'sentiment domain',
                                                          'sentiment language en', 'sentiment language de',
                                                          'sentiment language fr', 'sentiment language jp'])
    args = parser.parse_args()
    return args


def get_trainer(h_params: Box, model_type=None):
    if model_type == 'generator':
        num_epochs = h_params.num_decoder_epochs
        call_back = ModelCheckpoint(mode="min", save_top_k=1, monitor="avg_dev_loss",
                                    filename='{epoch:02d}-{dev_loss:.2f}.hdf5-{dev_metric:.2f}.hdf5')
    else:
        num_epochs = h_params.num_classification_epochs
        call_back = ModelCheckpoint(mode="max", save_top_k=1, monitor="dev_metric",
                                    filename='{epoch:02d}-{dev_loss:.2f}.hdf5-{dev_metric:.2f}.hdf5')
    trainer = Trainer(benchmark=False, deterministic=True, gpus=h_params.gpus,
                      num_sanity_val_steps=5, min_epochs=num_epochs, callbacks=[LoggingCallback(), call_back],
                      max_epochs=num_epochs,
                      default_root_dir=h_params.checkpoint_dir_path,
                      accumulate_grad_batches=h_params.gradient_accumulation_steps)
    return trainer


def main() -> None:
    """
    the main function, reads the arguments from the command line and the config files. Runs a single model on a single
    domain
    """
    args = define_parser()
    print(args)
    h_params = Box.from_yaml(filename=f'src/config files/config - {args.task_name}.yaml')
    h_params.source_domains = get_source(args.task_name, args.target, h_params.data_dir)
    h_params.target_domain = args.target

    model_prefix = MODEL_PREFIXES[args.model_type]
    if args.model_type == 'generator':
        print('generator')
        ds_path = f"{h_params.results_dir_path}/Datasets"
        if os.path.exists(ds_path):
            if not OVERRIDE_GENERATOR:
                exit(12)
            ds_files = [f'{ds_path}/{file}' for file in os.listdir(ds_path) if args.target in file]
            for f in ds_files:
                os.remove(f)
    h_params.results_dir_path = f"{h_params.results_dir_path}/{model_prefix}{h_params.model_name}"
    h_params.checkpoint_dir_path = f"{h_params.checkpoint_dir_path}/" \
                                   f"{model_prefix}{h_params.model_name}/{h_params.target_domain}"
    os.makedirs(h_params.results_dir_path, exist_ok=True)

    os.makedirs(h_params.checkpoint_dir_path, exist_ok=True)
    set_seed(h_params.random_seed)
    model_class = MODEL_CLASSES[args.model_type]

    trainer = get_trainer(h_params=h_params, model_type=args.model_type)
    model = model_class(h_params, test_domain=None)

    trainer.fit(model)
    if args.model_type == 'generator':
        print('creating datasets')
        create_signatures_datasets(h_params=h_params)
        return
    model.end_of_training()
    model_test(h_params, model_class, trainer, args.target)


def model_test(h_params, model_class, trainer, target=None):
    def merge_with_prev_results(source):
        metric = h_params.metric
        results["source"] = source
        full_results = {0: results}

        full_results = pd.DataFrame.from_dict(full_results, orient="index")
        try:
            prev_results = pd.read_csv(f"{h_params.results_dir_path}/{metric}.csv", index_col=0)
        except FileNotFoundError:
            prev_results = pd.DataFrame()

        full_results = full_results.append(prev_results, ignore_index=True)
        full_results.to_csv(f"{h_params.results_dir_path}/{metric}.csv")

    logs_folder = f"{h_params.checkpoint_dir_path}/lightning_logs/"
    model_loc = get_last_version(logs_folder)
    model_params = {'h_params': h_params, 'test_domain': 'dev'}
    model = model_class.load_from_checkpoint(model_loc, **model_params)
    trainer.test(model, test_dataloaders=model.data_loaders["dev"])
    dev_result = model.cur_metric
    trainer.test(model, test_dataloaders=model.data_loaders[f"test_{target}"])
    test_results = model.cur_metric
    results = {'dev': dev_result, target: test_results}
    merge_with_prev_results(h_params.source_domains)


def create_signatures_datasets(h_params):
    from modeling import SignaturesGenerator
    logs_folder = f"{h_params.checkpoint_dir_path}/lightning_logs/"
    model_loc = get_last_version(logs_folder)
    model_params = {'h_params': h_params, 'test_domain': 'dev'}
    base_model = SignaturesGenerator.load_from_checkpoint(model_loc, **model_params)

    base_dir, _ = os.path.split(h_params.results_dir_path)
    # set_seed(h_params.random_seed)
    base_model.h_params.results_dir_path = f"{base_dir}/Datasets"

    os.makedirs(base_model.h_params.results_dir_path, exist_ok=True)
    from modeling.hyper_drf import create_dataset
    create_dataset(base_model, mode='train')
    create_dataset(base_model, mode='dev')
    create_dataset(base_model, mode='test')


if __name__ == '__main__':
    main()
