from torch import nn

import logging
import torch
from pytorch_lightning import LightningModule, Callback
from torch import LongTensor, FloatTensor
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Any, Tuple, Union
import re
import pickle
import pandas as pd
from transformers import T5TokenizerFast, MT5TokenizerFast
from collections import defaultdict
import os


def create_signature_from_words(words: List[str]) -> str:
    """
    gets a list of words and creates a signature from them by adding commas
    Args:
        words: the words to merge
    Returns: the drf signature
    """
    if len(words) > 1:
        words = words[0] + " - " + ", ".join(words[1:])
    else:
        words = ", ".join(words)
    return words


def remove_list_duplicates(org_list: List[List[Any]]) -> List[List[Any]]:
    """
    makes each list unique while keeping the order of the first appearance
    Args:
        org_list: a list of lists

    Returns:
        the unique items of the list
    """
    ans = []
    for org in org_list:
        clean_list = []
        for item in org:
            if item not in clean_list:
                clean_list.append(item)
        ans.append(clean_list)
    return ans


def remove_extra_ids_from_txt(txt_list: List[str]) -> List[List[str]]:
    """
    removes the extra_ids token from a list of strings and splits them to words
    Args:
        txt_list: a list of strings
    Returns:
        a list of lists, each containing the words that are not extra_ids from each sentence
    """
    clean_txt = []
    for sign in txt_list:
        sign = re.sub('<extra_id_.>', '', sign)
        sign = re.sub('<extra_id_..>', '', sign)
        sign = sign.replace('  ', ' ').strip()

        sign = sign.split()
        clean_txt.append(sign)
    return clean_txt


def read_single_file_mnli_txt(path) -> List[str]:
    path = path.replace(".review", "")
    with open(path, "rb") as f:
        file = pickle.load(f)
    sentences, labels = file
    return sentences


def read_single_file_sentiment_txt(path) -> List[str]:
    file = pd.read_csv(path)
    sentences = file["text"].tolist()
    return sentences


def read_single_file_mnli(path: str, tokenizer: Union[T5TokenizerFast, MT5TokenizerFast], max_len: int) -> \
        Tuple[LongTensor, LongTensor, LongTensor]:
    """
    reads a file of the MNLI task and returns the text and labels of it
    Args:
        path: the path to the file
        tokenizer: the tokenizer to use for the tokenization
        max_len: the maximum sequence length
    Returns:
        The input ids, attention masks and labels of the sentences after tokenization
    """
    path = path.replace(".review", "")
    with open(path, "rb") as f:
        file = pickle.load(f)
    sentences, labels = file
    # sentences, labels = sentences[:10], labels[:10]
    # print('USING A SMALL DS!!!!')
    sentences = \
        tokenizer.batch_encode_plus(sentences, max_length=max_len, padding="max_length",
                                    return_tensors="pt", truncation=True, add_special_tokens=True)
    ids = sentences.data["input_ids"]
    masks = sentences.data["attention_mask"]

    labels = LongTensor(labels)
    return ids, masks, labels


def read_single_file_sentiment(path: str, tokenizer: Union[T5TokenizerFast, MT5TokenizerFast], max_len: int) -> \
        Tuple[LongTensor, LongTensor, LongTensor]:
    """
    reads a file of the sentiment task and returns the text and labels of it
    Args:
        path: the path to the file
        tokenizer: the tokenizer to use for the tokenization
        max_len: the maximum sequence length
    Returns:
        The input ids, attention masks and labels of the sentences after tokenization
    """
    file = pd.read_csv(path)
    sentences, tmp_labels = file["text"].tolist(), file["rating"].tolist()
    labels = []
    for label in tmp_labels:
        if label > 3:
            labels.append(1)
        else:
            labels.append(0)
    sentences = \
        tokenizer.batch_encode_plus(sentences, max_length=max_len, padding="max_length",
                                    return_tensors="pt", truncation=True, add_special_tokens=True)
    ids = sentences.data["input_ids"]
    masks = sentences.data["attention_mask"]

    labels = [int(label) for label in labels]
    labels = LongTensor(labels)
    return ids, masks, labels


class DefaultDictKey(defaultdict):
    def __missing__(self, key):
        self[key] = key
        return self[key]


def get_task_name_from_dir(path):
    task_dir = os.path.split(path)[0]
    if os.path.isdir(task_dir):
        task_dir = os.path.split(task_dir)[1]
    return task_dir


class EvalModule:
    """
    An evaluation module
    """

    def __init__(self, metric: str):
        """
        Args:
            metric: the metric by which to evaluate the prediction. options are accuracy and macro F1
        """
        self.preds = {"train": [], "dev": [], "test": []}
        self.truth = {"train": [], "dev": [], "test": []}
        if metric not in ['accuracy', "macro f1"]:
            raise NotImplementedError(f'metric {metric} not supported. Possible metrics are macro f1 and accuracy')
        self.metric = metric

    def add_batch(self, preds: LongTensor, truth: LongTensor, split) -> None:
        """
        adds the predictions of a single batch to the previous
        Args:
            preds: the current predictions
            truth: the current ground truth results
            split: which split the results are from - dev, train or test (could be any hashable object)
        """
        preds = preds.view(-1)
        truth = truth.view(-1)
        preds = preds.tolist()
        truth = truth.tolist()
        self.preds[split] += preds
        self.truth[split] += truth

    def compute(self, split) -> float:
        """
        computes and returns the metric between the predictions and the ground truth of the given split
        Args:
            split: the split for which the metric should be calculated

        Returns:
            A dictionary containing the metric name and it's value

        Notes:
            the function does not remove the previous predictions and the ground truth
        """
        metric = -1
        if self.metric == 'accuracy':
            metric = accuracy_score(self.truth[split], self.preds[split])
        elif self.metric == 'macro f1':
            metric = f1_score(self.truth[split], self.preds[split], average='macro')
        self.clear_results(split)
        return metric

    def clear_results(self, split) -> None:
        if "*" in split:
            for sp in self.preds.keys():
                self.truth[sp] = []
                self.preds[sp] = []
        else:
            self.truth[split] = []
            self.preds[split] = []


class HyperLinear(nn.Module):
    """
    The Hyper Network module
    """

    def __init__(self, base_model_hidden, base_model_emb, num_classes=2):
        """
        Args:
            base_model_hidden: the hidden dimension of the main model
            base_model_emb: the embedding dimension of the main model.
            num_classes: the number of classes to which we need to classify
        """
        super().__init__()
        self.in_dim = base_model_emb
        self.hidden_dim = base_model_emb
        self.out_common = base_model_emb
        self.num_classes = num_classes
        self.common = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_common),
            nn.ReLU(),
        )
        self.weight_head = nn.Linear(self.out_common, num_classes * base_model_hidden)
        self.bias_head = nn.Linear(self.out_common, num_classes)

    def forward(self, data: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        the Hyper Network forward call
        Args:
            data: the embedding of the hyper network's input

        Returns:
            the weight and bias of the classifier. the weight is of shape (batch X models hidden dim X num classes)
            and then the bias of shape (batch X num classes)
        """
        features = self.common(data)
        batch_size = data.shape[0]
        weight = self.weight_head(features)
        weight = weight.view(batch_size, -1, self.num_classes)
        bias = self.bias_head(features)
        bias = bias.view(batch_size, self.num_classes)
        return weight, bias


class LoggingCallback(Callback):
    """
    a simple logging module
    """

    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using GPU: {torch.cuda.is_available()}")

    def on_validation_end(self, trainer, pl_module: LightningModule) -> None:
        """
        called automatically at the end of each validation epoch
        Args:
            trainer:
            pl_module:

        Returns:
            None
        """
        self.logger.info("***** Validation results *****")
        metrics = filter(lambda x: x[0] not in ["log", "progress_bar"], trainer.callback_metrics.items())
        # Log results
        for key, metric in sorted(metrics):
            self.logger.info(f"{key} = {metric:.03f}\n")

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:
        """
        called automatically at the end of each validation epoch
        Args:
            trainer:
            pl_module:

        Returns:
            None
        """
        self.logger.info("***** Test results *****")
        self.logger.info(f"Num Training Epochs: {trainer.max_epochs}")
        # Log and save results to file
        metrics = filter(lambda x: x[0] not in ["log", "progress_bar"], trainer.callback_metrics.items())
        for key, metric in sorted(metrics):
            self.logger.info(f"{key} = {metric:.03f}\n")
