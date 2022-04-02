from .find_drf import find_drfs
from nltk.corpus import stopwords
from string import punctuation
import nltk
from sklearn.metrics import pairwise_distances
import string

import json
import torch
import pandas as pd
from pytorch_lightning import LightningModule
from torch import LongTensor, FloatTensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Any, Dict
from transformers import (
    BertTokenizerFast,
    BertModel,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)
from pytorch_pretrained_bert.optimization import BertAdam
from .model_utils import (
    EvalModule, create_signature_from_words, remove_list_duplicates, remove_extra_ids_from_txt, DefaultDictKey,
    read_single_file_mnli_txt, read_single_file_sentiment_txt, read_single_file_mnli, read_single_file_sentiment,
    get_task_name_from_dir
)

nltk.download('stopwords')


class SignaturesGenerator(LightningModule):
    LOSS_IGNORE_ID = -100

    def remove_stop_words(self, org_list: List[List[str]]) -> List[List[str]]:
        """
        removes stopwords and punctuation from all the given sentences
        Args:
            org_list: a list of sentences, where each is split into words

        Returns:
            a list of sentences split into words without stopwords or punctuation
        """
        stop_words = self.get_stop_words()
        ans = []
        table = str.maketrans(dict.fromkeys(string.punctuation))
        for org in org_list:
            clean_list = []
            for item in org:
                item = item.translate(table)
                if item not in stop_words:
                    clean_list.append(item)
            ans.append(clean_list)
        return ans

    def clean_generated_txt(self, signature: List[str]) -> List[str]:
        """
        gets a list of sentences and creates a DRF signature from each
        Args:
            signature: a list of generated texts

        Returns:
            a list of the clean texts - without stopwords and extra ids and in the correct format with punctuation
        """
        signature = remove_extra_ids_from_txt(signature)
        signature = remove_list_duplicates(signature)
        signature = self.remove_stop_words(signature)
        ans = []
        for sign in signature:
            sign = create_signature_from_words(words=sign)
            ans.append(sign)
        signature = ans
        return signature

    def _init_models(self) -> None:
        """
        initializes the Transformers and tokenizers required for the model

        """
        if self.is_multi_lang:
            print("MT5 model")
            self.model = MT5ForConditionalGeneration.from_pretrained(self.h_params.model_path)
            self.model: MT5ForConditionalGeneration
            self.tokenizer = MT5Tokenizer.from_pretrained(self.h_params.model_path)
            self.tokenizer: MT5Tokenizer
            self.out_dim = self.model.encoder.config.d_model
            bert_path = "bert-base-multilingual-uncased"
        else:
            print("T5 model")
            self.model = T5ForConditionalGeneration.from_pretrained(self.h_params.model_path)
            self.model: T5ForConditionalGeneration
            self.tokenizer = T5TokenizerFast.from_pretrained(self.h_params.model_path)
            self.tokenizer: T5TokenizerFast
            self.out_dim = self.model.encoder.config.d_model
            bert_path = "bert-base-uncased"
        device = 'cpu'
        if self.h_params.gpus > 0:
            device = 'cuda:0'
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.embedding = BertModel.from_pretrained(bert_path).embeddings.word_embeddings
        self.emb_dim = self.bert.config.hidden_size
        self.bert.to(device)
        self.embedding.to(device)

    def __init__(self, h_params, test_domain):
        """
        Args:
            h_params: the hyper parameters
            test_domain: the domain we are currently testing
        """
        self.drfs = None
        self.drfs_embedding = {}
        self.num_tokens_in_hyper = 20
        super().__init__()
        self.test_domain = test_domain
        self.src_txt = f'to {h_params.target_domain}'
        self.h_params = h_params
        self.task_name = get_task_name_from_dir(self.h_params.data_dir)
        if not (self.task_name == 'mnli' or self.task_name == 'sentiment' or 'sentiment language' in self.task_name):
            raise NotImplementedError(f'No such task {self.task_name}.'
                                      f' Options are mnli, sentiment, or sentiment language')
        self.is_multi_lang = h_params.is_multi_lang
        self.metric = h_params.metric
        self.h_params.to_yaml(filename=f"{self.h_params.results_dir_path}/{self.src_txt} parameters.txt")
        self._init_models()
        self.num_classes = self.h_params.num_classes
        self.data_loaders = self._init_data_loaders()
        self.loss_fn = CrossEntropyLoss(ignore_index=SignaturesGenerator.LOSS_IGNORE_ID)
        self.eval_metric_scorer = EvalModule(self.metric)
        self.eval_predictions = dict()
        self.metric_per_epoch = pd.DataFrame(columns=["dev"])
        self.cur_metric = -1
        self.sentences = []
        self.test_sentences_out = []
        self.test_generated_drfs = []
        self.test_selected_drfs = []
        self.loss_per_epoch = pd.DataFrame(columns=["dev"])

    def _init_data_loaders(self) -> Dict[str, DataLoader]:
        train_data_loader = self._create_data_loader_train_dev(True)
        dev_data_loader = self._create_data_loader_train_dev(False)
        test_data_loaders = self._create_data_loader_test()
        test_data_loaders["train"] = train_data_loader
        test_data_loaders["dev"] = dev_data_loader
        return test_data_loaders

    def _create_data_loader_train_dev(self, is_train: bool) -> DataLoader:
        if self.test_domain is not None:
            return None
        ids, masks, signature = LongTensor(), LongTensor(), LongTensor()
        mode = "train" if is_train else "dev"
        for domain in self.h_params.source_domains:
            print(f"{mode} {domain}")
            cur_file_path = self.h_params.data_dir + domain + f"/{mode}.review"
            cur_ids, cur_masks, cur_labels = \
                self.read_single_file(cur_file_path)
            ids = torch.cat([ids, cur_ids])
            sign = self.get_all_signatures(cur_ids, domain)
            masks = torch.cat([masks, cur_masks])
            signature = torch.cat([signature, sign])
        dataset = TensorDataset(ids, masks, signature)
        data_loader = DataLoader(dataset, batch_size=self.h_params.batch_size, shuffle=is_train)
        return data_loader

    def _create_data_loader_test(self) -> Dict[str, DataLoader]:
        return {}

    def read_single_file(self, path) -> Tuple[LongTensor, LongTensor, LongTensor]:
        """
        reads a file and returns the text and labels of it
        Args:
            path: the path to the file
        Returns:
            The input ids, attention masks and labels of the sentences after tokenization
        """
        if self.task_name == 'mnli':
            return read_single_file_mnli(path, self.tokenizer, self.h_params.max_seq_length)
        elif 'sentiment' in self.task_name:
            return read_single_file_sentiment(path, self.tokenizer, self.h_params.max_seq_length)

    def get_all_signatures(self, sentences: LongTensor, domain: str) -> LongTensor:
        """
        gets the annotated DRF signature for each sentence
        Args:
            sentences: a Tensor containing the ids of each sentence
            domain: the domain from which the sentences are from

        Returns:
            a Tensor containing the tokens of the annotated DRF signature
        """
        t5_tokens = []
        for idx, sen in enumerate(sentences):
            cur_txt = self.get_single_signature(sen, domain)
            if idx == 0:
                print(cur_txt)
            cur_tokens = self.tokenizer.encode(cur_txt, padding="max_length", truncation=True,
                                               max_length=self.num_tokens_in_hyper, return_tensors="pt").squeeze(0)
            t5_tokens.append(cur_tokens)
        t5_tokens = torch.stack(t5_tokens)
        return t5_tokens

    def get_single_signature(self, sentence: LongTensor, domain: str) -> str:
        """
        matches the sentence with the DRFs closest to it and creates a DRF signature from it
        Args:
            sentence: the sentence to find the DRF signature for
            domain: the domain from which this sentence is taken

        Returns:
            a DRF signature
        """
        device = self.embedding.weight.device

        def match_single_signature_to_sentence(sen: str):
            sentence_emb = self.embedding(self.bert_tokenizer(sen, return_tensors='pt', add_special_tokens=False)
                                          ['input_ids'].to(device)).squeeze(0).cpu()
            sign_emb = self.drfs_embedding[domain]
            pairwise_dist = pairwise_distances(sentence_emb, sign_emb, metric='minkowski')
            per_word_min_dist = pairwise_dist.min(axis=0)
            sorted_words_min_distances = sorted(range(len(per_word_min_dist)), key=per_word_min_dist.__getitem__)
            closest_word_ids = sorted_words_min_distances[: self.h_params.num_drfs_per_sen]
            drf_list = self.drfs[domain]
            closest_words = [drf_list[drf_id] for drf_id in closest_word_ids]
            return closest_words

        if self.drfs is None:
            self.get_drfs_set()
        lang_dom = self.get_domain_string(domain)
        rel_drfs = []
        sentence = self.tokenizer.decode(sentence, skip_special_tokens=True)
        with torch.no_grad():
            if domain in self.h_params.source_domains:
                rel_drfs = match_single_signature_to_sentence(sentence)
        if self.is_multi_lang:
            word_list = lang_dom + rel_drfs
            word_list = [f"<extra_id_{i}> {word} " for i, word in enumerate(word_list)]
            words = "<pad>" + "".join(word_list)
        else:
            words = " ".join(lang_dom) + ": " + ", ".join(rel_drfs)
        return words

    def get_domain_dict(self) -> Dict[str, str]:
        """
        gets a domain with the full string representing each domain
        Returns:
        A dictionary (could be a defaultdict) whose keys are the domains and the values are the strings
        """
        if self.task_name == 'sentiment':
            dom_to_lang = {'en-dvd': 'dvd', 'en-music': 'music', 'en-books': 'books',
                           'de-dvd': 'dvd', 'de-music': 'musik', 'de-books': 'bücher',
                           'fr-dvd': 'dvd', 'fr-music': 'musique', 'fr-books': 'livres',
                           'jp-dvd': 'dvd', 'jp-music': '音楽', 'jp-books': '本'}
            return dom_to_lang
        dom_to_words = DefaultDictKey(lambda x: x)
        return dom_to_words

    def get_domain_string(self, domain: str) -> List[str]:
        """

        Args:
            domain: the domain from which the sentence arrived
        Returns:
            the domain part of the DRF signature
        """
        dom_to_words = self.get_domain_dict()
        words = [dom_to_words[domain]]
        return words

    def get_drfs_set(self) -> None:
        drfs = find_drfs(self)
        self.drfs = {}
        for domain in drfs:
            drfs_emb = []
            self.drfs[domain] = []
            with torch.no_grad():
                for drf in drfs[domain]:
                    drf_bert_tokens = self.bert_tokenizer.batch_encode_plus([drf], return_tensors="pt",
                                                                            add_special_tokens=False)["input_ids"]
                    drf_tokens = self.tokenizer.batch_encode_plus([drf], return_tensors="pt",
                                                                  add_special_tokens=False)["input_ids"]
                    if drf_tokens.shape[1] > 3 or drf_bert_tokens.shape[1] > 3:
                        continue
                    self.drfs[domain].append(drf)
                    device = self.embedding.weight.device
                    drf_emb = self.embedding(drf_bert_tokens.to(device)).mean(axis=1).squeeze().cpu()
                    drfs_emb.append(drf_emb)
            drfs_emb = torch.stack(drfs_emb).numpy()
            self.drfs_embedding[domain] = drfs_emb

    def read_single_file_txt(self, path) -> List[str]:
        """
        reads a file and returns only the sentences it contains
        Args:
            path: the path to the file
        Returns:
            the sentences in that file
        """
        if self.task_name == 'mnli':
            return read_single_file_mnli_txt(path)
        elif 'sentiment' in self.task_name:
            return read_single_file_sentiment_txt(path)

    def get_stop_words(self) -> List[str]:
        """
        Returns: a list of all the stopwords in the language the model is in
        """
        if self.is_multi_lang:
            lang_to_full = {"de": "deutsch", "en": "english", "jp": "japanese", "fr": "french"}
            all_stopwords = set()
            if self.task_name == 'sentiment':
                languages = [domain.split('-')[0] for domain in self.h_params.source_domains]
            else:
                lang = self.task_name.split('language')[1].strip()
                languages = [lang]
            for lang in languages:
                lang = lang_to_full[lang]
                try:
                    cur_stop = set(stopwords.words(lang))
                except OSError:
                    continue
                all_stopwords = all_stopwords.union(cur_stop)
            all_stopwords = sorted(list(all_stopwords))
        else:
            all_stopwords = sorted(list(stopwords.words('english')))
        all_stopwords += list(punctuation)
        return all_stopwords

    def get_representation_tensor_single_sentence(self, words: List[str]) -> FloatTensor:
        """
        calculates the Tensor representation of the DRF signature
        Args:
            words: the words to be in the DRF signature
        Returns:
            a Tensor with the CLS token embedding of the words
        """
        words = create_signature_from_words(words)
        encoded = self.bert_tokenizer.encode_plus(words, truncation=True, max_length=self.num_tokens_in_hyper,
                                                  padding="max_length", return_tensors="pt",
                                                  add_special_tokens=True)
        words, _ = encoded["input_ids"], encoded["attention_mask"][0]
        with torch.no_grad():
            self.bert.eval()
            embedding = self.bert(words.to(self.bert.device))['pooler_output'].to('cpu').squeeze(0)
        return embedding

    def classify_item(self, encoder_outputs, in_hyper):
        """
        passes a sentence through the HN and calculates the logits
        Args:
            encoder_outputs: the sentence representation from the encoder
            in_hyper: the metadata to be inserted to the HN
        Returns:
            logits - a tensor of shape (batch X number of classes)
        """
        batch_size = in_hyper.shape[0]
        encoder_outputs = encoder_outputs.mean(dim=1)
        encoder_outputs = encoder_outputs.unsqueeze(1)
        weight, bias = self.hyper(in_hyper)
        encoder_outputs = encoder_outputs.to(self.device)
        temp = torch.bmm(encoder_outputs, weight)
        temp = temp.view(batch_size, -1)
        logits = temp + bias
        return logits

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, signature: LongTensor) -> FloatTensor:
        """
        the forward pass of the model
        Args:
            input_ids: the input ids of the batch. a tensor of shape (batch, seq_length)
            attention_mask: the attention masks of the batch to avoid doing attention with pad tokens.
             a tensor of shape (batch, seq_length)
            signature: the input ids of the annotated signature to be used as a gold label for the generation
        Returns:
            the loss between the generated and the gold label texts
        """
        gen_return = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=signature)
        return gen_return[0]

    def _step(self, batch: Tuple[LongTensor, LongTensor, LongTensor], mode: str) -> FloatTensor:
        """
        Performs a training step.

        Args:
            batch: a tuple of tensors that contains the input_ids, the attention masks
            and the signatures' input ids, each of shape (batch, seq_length)
        Returns:
            tuple:
                Tensor of shape (1,) with the loss after the forward pass.
        """
        input_ids, attention_mask, signature = batch
        loss = self(input_ids=input_ids, attention_mask=attention_mask, signature=signature)
        if self.h_params.gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        return loss

    def training_step(self, batch: Tuple, batch_idx) -> FloatTensor:
        """
        Compute and return the training loss.

        Args:
            batch: a tuple with the data of a single batch as described in _step()
            batch_idx: ignored
        Returns:
            Tensor of shape (1,) with the loss after the forward pass.
        """
        loss = self._step(batch, 'train')
        self.log("train_loss", float(loss), on_step=False, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def _eval_step(self, batch: Tuple, mode) -> FloatTensor:
        """
        Compute and return the evaluation loss.

        Args:
            batch: a tuple with the data of a single batch as described in _step()
            mode: whether it is a validation or test batch
        Returns:
            Tensor of shape (1,) with the loss after the forward pass.
        """
        return self._step(batch, mode)

    def validation_step(self, batch: Tuple, batch_idx) -> FloatTensor:
        """
        Compute and return the validation loss.

        Args:
            batch: a tuple with the data of a single batch as described in _step()
            batch_idx: ignored
        Returns:
            Tensor of shape (1,) with the loss after the forward pass.
        """
        return self._eval_step(batch, 'dev')

    def test_step(self, batch: Tuple, batch_idx) -> FloatTensor:
        """
        Compute and return the test loss.

        Args:
            batch: a tuple with the data of a single batch as described in _step()
            batch_idx: ignored
        Returns:
            Tensor of shape (1,) with the loss after the forward pass.
        """
        self.sentences += self.tokenizer.batch_decode(batch[0], skip_special_tokens=True)
        return self._eval_step(batch, 'test')

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the training epoch with the outputs of all training steps.

        Args:
            outputs: List of outputs like defined in training_step(), or if there are multiple data_loaders,
                a list containing a list of outputs for each dataloader.
        """
        losses = [cur_out["loss"] for cur_out in outputs]
        avg_epoch_loss = torch.stack(losses).mean()
        self.log(f"avg_train_loss", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _eval_epoch_end(self, outputs: List[Any], split: str) -> None:
        """
        calculates the loss and the metric at the end of each validation or test epoch
        Args:
            outputs: List of outputs like defined in training_step(), or if there are multiple data_loaders,
                a list containing a list of outputs for each dataloader.
            split: what split are we using - dev or test
        """
        losses = [cur_out for cur_out in outputs]
        avg_epoch_loss = torch.stack(losses).mean()
        self.log(f"avg_{split}_loss", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        epoch_score = self.eval_metric_scorer.compute(split)
        self.log(f"{split}_{self.metric}", epoch_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{split}_metric", epoch_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if split == "dev":
            self.metric_per_epoch = self.metric_per_epoch.append({split: epoch_score}, ignore_index=True)
            self.cur_metric = epoch_score
        if "test" in split:
            self.cur_metric = epoch_score
        self.test_sentences_out, self.test_generated_drfs, self.test_selected_drfs = [], [], []
        losses = [cur_out for cur_out in outputs]
        avg_epoch_loss = float(torch.stack(losses).mean().cpu())
        self.loss_per_epoch = self.loss_per_epoch.append({split: avg_epoch_loss}, ignore_index=True)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the validation epoch with the outputs of all steps. saves the loss at the end of each
        epoch to a file

        Args:
            outputs: List of outputs like defined in validation_step(), or if there are multiple data_loaders,
                a list containing a list of outputs for each dataloader.
        """
        self._eval_epoch_end(outputs, "dev")
        self.loss_per_epoch.to_csv(f"{self.h_params.results_dir_path}/{self.src_txt} avg test loss per epoch.csv")

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the test epoch with the outputs of all steps.

        Args:
            outputs: List of outputs like defined in test_step(), or if there are multiple data_loaders,
                a list containing a list of outputs for each dataloader.
        """
        self.write_eval_predictions()
        self._eval_epoch_end(outputs, "test")

    def write_eval_predictions(self) -> None:
        """
        saves the predictions of the model in the current epoch
        """
        mode_txt = 'domain ' + self.test_domain
        out_path = f"{self.h_params.results_dir_path}/{self.src_txt} {mode_txt} predictions.json"
        sentences, preds, truth = \
            self.sentences, self.eval_metric_scorer.preds['test'], self.eval_metric_scorer.truth['test']
        data = [{'sentence': sen, 'truth': label, 'pred': pred} for sen, pred, label in zip(sentences, preds, truth)]
        with open(out_path, 'w') as json_file:
            json.dump(data, json_file)
        self.sentences = []

    def end_of_training(self):
        """
        saves the metric from each epoch to a file
        """
        self.metric_per_epoch.to_csv(f"{self.h_params.results_dir_path}/{self.src_txt} {self.metric} per epoch.csv")

    def train_dataloader(self) -> DataLoader:
        return self.data_loaders["train"]

    def val_dataloader(self) -> DataLoader:
        return self.data_loaders["dev"]

    def domain_test_dataloader(self, domain) -> DataLoader:
        return self.data_loaders[f"test_{domain}"]

    def configure_optimizers(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.trainer.max_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.h_params.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        t_total = (
                (len(self.data_loaders["train"].dataset) // (self.h_params.batch_size *
                                                             float(max(1, self.h_params.gpus))))
                // self.h_params.gradient_accumulation_steps * float(num_epochs)
        )
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=float(self.h_params.learning_rate),
                             warmup=self.h_params.warmup_proportion,
                             t_total=t_total)
        return {"optimizer": optimizer}

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
