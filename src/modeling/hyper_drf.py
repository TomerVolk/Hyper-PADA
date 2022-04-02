from .signatures_generator import SignaturesGenerator
from typing import List, Any, Dict, Tuple

from torch import LongTensor, FloatTensor
import torch
from transformers import T5ForConditionalGeneration
from torch.utils.data import TensorDataset, DataLoader
from .model_utils import HyperLinear
import json
import os
import numpy as np


def generate(model: T5ForConditionalGeneration, input_ids: LongTensor, attention_mask: LongTensor, h_params) -> \
        LongTensor:
    """
    generates the DRF signatures of a single batch of sentences
    Args:
        model: the model to use for the generation
        input_ids: the ids of the input sentence. should be of shape (batch, seq_length)
        attention_mask: the attention masks of the batch to avoid doing attention with pad tokens.
             a tensor of shape (batch, seq_length)
        h_params: the hyper parameters object
    Returns:
        the ids of the generated DRF signatures of shape (batch, num repetitions, seq_length)
    """
    num_repetitions = h_params.num_repetitions
    signature = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_return_sequences=num_repetitions,
                               diversity_penalty=h_params.diversity_penalty, num_beams=max(int(num_repetitions), 2),
                               num_beam_groups=max(int(num_repetitions), 2),
                               repetition_penalty=h_params.repetition_penalty,
                               length_penalty=h_params.length_penalty, early_stopping=h_params.early_stopping)
    return signature


def create_dataset(model: SignaturesGenerator, mode: str) -> None:
    """
    if the generations dataset doesn't exists, creates it.
    Args:
        model: the pretrained model from the first stage
        mode: what split to use: train, dev or test
    """
    if os.path.exists(f"{model.h_params.results_dir_path}/{model.src_txt} {mode} dataset.json"):
        print('Dataset Exists')
        return
    with torch.no_grad():
        max_len = 20
        h_params = model.h_params
        sentences, masks = torch.LongTensor(), torch.LongTensor()
        signatures = []
        if mode == 'train' or mode == 'dev':
            domains = h_params.source_domains
        else:
            domains = [h_params.target_domain]
        for domain in domains:
            print(f"{mode} {domain}")
            cur_file_path = h_params.data_dir + domain + f"/{mode}.review"
            cur_ids, cur_masks, _ = model.read_single_file(cur_file_path)
            cur_signatures = []
            for single_id, single_mask in zip(cur_ids, cur_masks):
                single_id = single_id.unsqueeze(0)
                single_mask = single_mask.unsqueeze(0)

                signature = generate(model=model.model, input_ids=single_id.to(model.device),
                                     attention_mask=single_mask.to(model.device), h_params=h_params)
                cur_signatures.append(signature)
            cur_signatures = [model.tokenizer.batch_decode(x) for x in cur_signatures]
            cur_signatures = [model.tokenizer.batch_encode_plus(x, max_length=max_len,
                                                                padding="max_length",
                                                                truncation=True,
                                                                add_special_tokens=True)['input_ids']
                              for x in cur_signatures]
            sentences = torch.cat([sentences, cur_ids])
            signatures += cur_signatures
            masks = torch.cat([masks, cur_masks])
        sentences = sentences.detach().clone().cpu().tolist()
        results = {'sentences': sentences, 'DRF signatures': signatures}
        json_object = json.dumps(results)

        with open(f"{model.h_params.results_dir_path}/{model.src_txt} {mode} dataset.json", "w") as outfile:
            outfile.write(json_object)
        sentences = model.tokenizer.batch_decode(sentences, skip_special_tokens=True)
        signatures = [model.tokenizer.batch_decode(pre, skip_special_tokens=True) for pre in signatures]
        temp = [{f'DRF signature {idx}': s for idx, s in enumerate(sign)} for sign in signatures]
        for t, sen in zip(temp, sentences):
            t['sentence'] = sen
        import pandas as pd
        temp_df = pd.DataFrame(temp)
        cols = list(temp_df.columns)
        cols.remove('sentence')
        cols = list(temp_df.columns)
        cols.remove('sentence')
        cols = ['sentence'] + cols
        temp_df = temp_df[cols]
        temp_df.to_csv(f"{model.h_params.results_dir_path}/{model.src_txt} {mode} dataset.csv")


class HyperDRF(SignaturesGenerator):

    def __init__(self, h_params, test_domain):
        """
        Args:
            h_params: the hyper parameters
            test_domain: the domain we are currently testing
        """
        super(HyperDRF, self).__init__(h_params, test_domain=test_domain)
        self.hyper = HyperLinear(base_model_hidden=self.out_dim, base_model_emb=self.emb_dim,
                                 num_classes=self.num_classes)

    def _create_data_loader_train_dev(self, is_train: bool) -> DataLoader:
        mode = 'train' if is_train else 'dev'
        return self._create_data_loader(mode)

    def _create_data_loader(self, split: str) -> DataLoader:
        dataset_path, _ = os.path.split(self.h_params.results_dir_path)
        if split in ['train', 'dev']:
            domains = self.h_params.source_domains
        else:
            domains = [self.h_params.target_domain]
        with open(f"{dataset_path}/Datasets/{self.src_txt} {split} dataset.json", 'rb') as f:
            generations = json.load(f)
        signatures = torch.from_numpy(np.array(generations['DRF signatures'], dtype=int)).type(torch.int)
        if split == 'train':
            signatures = signatures[:, 0, :].unsqueeze(1)

        labels, ids, masks = torch.LongTensor(), torch.LongTensor(), torch.LongTensor()
        for domain in domains:
            cur_file_path = self.h_params.data_dir + domain + f"/{split}.review"
            cur_ids, cur_masks, cur_labels = \
                self.read_single_file(cur_file_path)
            ids = torch.cat([ids, cur_ids])
            labels = torch.cat([labels, cur_labels])
            masks = torch.cat([masks, cur_masks])
        # ids, masks, signatures, labels = ids[:50], masks[:50], signatures[:50], labels[:50]
        # print('USING A SMALL DS!!!')
        ids, masks, _, domain_embeddings = \
            self.get_signature_representation(input_ids=ids, attention_mask=masks, signatures=signatures)
        dataset = TensorDataset(ids, masks, labels, domain_embeddings)
        data_loader = DataLoader(dataset, batch_size=self.h_params.batch_size, shuffle=split == 'train')
        return data_loader

    def _create_data_loader_test(self) -> Dict[str, DataLoader]:
        if self.test_domain is None:
            return {}
        data_loader = self._create_data_loader(split='test')
        data_loaders = {}
        domain = self.h_params.target_domain
        print(f"test {domain}")
        data_loaders[f"test_{domain}"] = data_loader
        return data_loaders

    def get_single_signature_representation(self, in_hyper: LongTensor) -> FloatTensor:
        """
        gets the input ids of a batch of DRF signatures and returns their embedding
        Args:
            in_hyper: the input ids of the DRF signatures. of shape (batch, seq_length)
        Returns: the CLS embedding of the DRF signature. of shape (batch, emb_dim)
        """
        generated_txt = self.tokenizer.batch_decode(in_hyper, skip_special_tokens=True, )
        in_hyper_txt = self.clean_generated_txt(generated_txt)
        representation_list = []
        for pre in in_hyper_txt:
            cur_in_hyper = self.get_representation_tensor_single_sentence([pre])
            representation_list.append(cur_in_hyper)
        in_hyper = torch.stack(representation_list).to(self.device)
        return in_hyper

    def get_signature_representation(self, input_ids: LongTensor, attention_mask: LongTensor, signatures: LongTensor)\
            -> Tuple[LongTensor, LongTensor, LongTensor, FloatTensor]:
        """
        gets the input ids of all the DRF signatures and returns their embedding
        Args:
            input_ids: the input ids of the sentences. a tensor of shape (num sentences, seq_length)
            attention_mask: the attention masks of the sentences to avoid doing attention with pad tokens.
             a tensor of shape (num sentences, seq_length)
            signatures: the ids of the generated DRF signatures.
             a tensor of shape (num sentences, num repetitions, seq_length)
        Returns:
            the ids of the sentences (shape (num sentences, seq_length)),
             their attention masks (shape (num sentences, seq_length)),
             the ids of the generated DRF signature (shape (num sentences, nup repetitions, seq_length))
              and their representations (shape (num sentences, nup repetitions, emb_dim))
        """
        device = 'cpu'
        if self.h_params.gpus > 0:
            device = 'cuda:0'
        tensors = []
        for idx, (pre, cur_ids, cur_mask) in enumerate(zip(signatures, input_ids, attention_mask)):
            pre = pre.to(device)
            cur_rep = self.get_single_signature_representation(in_hyper=pre).clone().detach().cpu()
            tensors.append(cur_rep)
        tensors = torch.stack(tensors)
        return input_ids, attention_mask, signatures, tensors

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the validation epoch with the outputs of all steps. saves the loss and metrics at
         the end of each epoch to a file

        Args:
            outputs: List of outputs like defined in validation_step(), or if there are multiple data_loaders,
                a list containing a list of outputs for each dataloader.
        """
        self._eval_epoch_end(outputs, "dev")
        self.end_of_training()

    def _step(self, batch: Tuple[LongTensor, LongTensor, LongTensor, FloatTensor], mode: str) -> FloatTensor:
        """
        Performs a training step.

        Args:
            batch: a tuple of tensors that contains the input_ids (shape (batch, seq_length))
            the attention masks (shape (batch, seq_length)), the sentences' labels (shape (batch,))
            and the DRF signatures' CLS embeddings (shape (batch, emb_dim))
        Returns:
            tuple:
                Tensor of shape (1,) with the loss after the forward pass.
        """
        input_ids, attention_mask, label_ids, signature = batch
        logits = self(input_ids=input_ids, attention_mask=attention_mask, signature=signature)
        loss = self.loss_fn(logits.view(-1, self.num_classes), label_ids.view(-1))
        preds = logits.detach().clone().cpu().argmax(dim=-1)
        self.eval_metric_scorer.add_batch(preds, label_ids.detach().cpu(), mode)
        if self.h_params.gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        return loss

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, signature: FloatTensor = None) -> FloatTensor:
        """
        the forward pass of the model
        Args:
            input_ids: the input ids of the batch. a tensor of shape (batch, seq_length)
            attention_mask: the attention masks of the batch to avoid doing attention with pad tokens.
             a tensor of shape (batch, seq_length)
            signature: the embedding of the generated DRF signature to be inserted to the HN.
            of shape (batch, num repetitions, emb_dim)
        Returns:
            the logits of the label predictions
        """
        encoder_outputs = self.model.encoder(input_ids, attention_mask=attention_mask)[0]
        batch, num_rep, emb_dim = signature.shape
        in_hyper = signature.view(batch * num_rep, emb_dim)
        encoder_outputs = torch.repeat_interleave(encoder_outputs, num_rep, dim=0)
        logits = self.classify_item(encoder_outputs, in_hyper)
        logits = logits.view(batch, num_rep, -1)
        logits = logits.mean(axis=1)
        return logits
