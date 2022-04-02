from .signatures_generator import SignaturesGenerator
from .model_utils import HyperLinear
import torch
from typing import Tuple, List, Any, Dict
import random
from torch.utils.data import TensorDataset, DataLoader
from torch import LongTensor, FloatTensor
from collections import defaultdict


class HyperDomainName(SignaturesGenerator):

    def __init__(self, h_params, test_domain):
        self.mode = None
        super(HyperDomainName, self).__init__(h_params, test_domain=test_domain)
        self.hyper = HyperLinear(base_model_hidden=self.out_dim, base_model_emb=self.emb_dim,
                                 num_classes=self.num_classes)

    def get_domain_dict(self) -> Dict[str, str]:
        """
        gets a domain with the full string representing each domain.
        Returns:
        A dictionary (could be a defaultdict) whose keys are the domains and the values are the strings
        """
        base_dict = super(HyperDomainName, self).get_domain_dict()

        if self.mode == 'train':
            prob = random.random()
            if prob < self.h_params.unk_domain_prob:
                base_dict = defaultdict(lambda: self.tokenizer.unk_token)
        elif self.mode != 'dev':
            base_dict = defaultdict(lambda: self.tokenizer.unk_token)
        return base_dict

    def get_hyper_all(self, num_sentences: int, domain: str) -> FloatTensor:
        """
        gets the HN input representation of all the sentences
        Args:
            num_sentences: the number of sentences from that domain
            domain: the domain from which the sentences are taken
        Returns: the CLS embedding of all the domain names of each sentence. a tensor of shape (num sen, emb_dim)
        """
        tensors = []
        for idx in range(num_sentences):
            cur_words = self.get_domain_string(domain)
            if idx < 2:
                print(cur_words)
            cur_rep = self.get_representation_tensor_single_sentence(cur_words)
            tensors.append(cur_rep)
        tensors = torch.stack(tensors)
        return tensors

    def _create_data_loader_train_dev(self, is_train: bool):
        self.num_tokens_in_hyper = 10
        ids, masks, labels, domain_embeddings = LongTensor(), LongTensor(), LongTensor(), FloatTensor()
        mode = "train" if is_train else "dev"
        self.mode = mode
        for domain in self.h_params.source_domains:
            print(f"{mode} {domain}")
            cur_file_path = self.h_params.data_dir + domain + f"/{mode}.review"
            cur_ids, cur_masks, cur_labels = \
                self.read_single_file(cur_file_path)
            ids = torch.cat([ids, cur_ids])
            labels = torch.cat([labels, cur_labels])
            cur_dom_emb = self.get_hyper_all(len(cur_ids), domain)
            masks = torch.cat([masks, cur_masks])
            domain_embeddings = torch.cat([domain_embeddings, cur_dom_emb])
        dataset = TensorDataset(ids, masks, labels, domain_embeddings)
        data_loader = DataLoader(dataset, batch_size=self.h_params.batch_size, shuffle=is_train)
        return data_loader

    def _create_data_loader_test(self):
        self.mode = 'test'
        domain = self.h_params.target_domain

        cur_file_path = self.h_params.data_dir + domain + "/test.review"
        ids, masks, labels = \
            self.read_single_file(cur_file_path)
        domain_embeddings = self.get_hyper_all(len(ids), domain)

        dataset = TensorDataset(ids, masks, labels, domain_embeddings)
        data_loader = DataLoader(dataset, batch_size=self.h_params.batch_size, shuffle=False)
        data_loaders = {f"test_{domain}": data_loader}
        return data_loaders

    def _step(self, batch: Tuple[LongTensor, LongTensor, LongTensor, FloatTensor], mode: str) -> FloatTensor:
        """
        Performs a training step.

        Args:
            batch: a tuple of tensors that contains the input_ids (shape (batch, seq_length))
            the attention masks (shape (batch, seq_length)), the sentences' labels (shape (batch,))
            and the domains' CLS embeddings (shape (batch, emb_dim))
        Returns:
            tuple:
                Tensor of shape (1,) with the loss after the forward pass.
        """
        input_ids, attention_mask, label_ids, signature = batch
        logits = self(input_ids=input_ids, attention_mask=attention_mask, signature=signature)
        loss = self.loss_fn(logits.view(-1, self.num_classes), label_ids.view(-1))
        preds = logits.detach().cpu().argmax(dim=-1)
        if mode != "train":
            self.eval_metric_scorer.add_batch(preds, label_ids.detach().cpu(), mode)
        if self.h_params.gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        return loss

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor = None, signature: FloatTensor = None) ->\
            FloatTensor:
        """
        the forward pass of the model
        Args:
            input_ids: the input ids of the batch. a tensor of shape (batch, seq_length)
            attention_mask: the attention masks of the batch to avoid doing attention with pad tokens.
             a tensor of shape (batch, seq_length)
            signature: the embedding of the domain to be inserted to the HN.
            of shape (batch, num repetitions, emb_dim)
        Returns:
            the logits of the label predictions
        """
        encoder_outputs = self.model.encoder(input_ids, attention_mask=attention_mask)[0]
        logits = self.classify_item(encoder_outputs, signature)
        return logits

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the validation epoch with the outputs of all steps. saves the loss and metrics
        at the end of each epoch to a file

        Args:
            outputs: List of outputs like defined in validation_step(), or if there are multiple data_loaders,
                a list containing a list of outputs for each dataloader.
        """
        self._eval_epoch_end(outputs, "dev")
        self.end_of_training()
