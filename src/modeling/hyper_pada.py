import torch
from torch import LongTensor, FloatTensor
from .hyper_drf import HyperDRF
from typing import Tuple


class HyperPada(HyperDRF):

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
            the ids of the sentences (shape (num sentences, num repetitions, seq_length)),
             their attention masks (shape (num sentences, num repetitions, seq_length)),
             the ids of the generated DRF signatures (shape (num sentences, nup repetitions, seq_length))
              and their representations (shape (num sentences, nup repetitions, emb_dim))
        """
        input_ids, attention_mask, signatures, in_hyper = \
            super(HyperPada, self).get_signature_representation(
                input_ids=input_ids, attention_mask=attention_mask, signatures=signatures)
        input_ids, attention_mask, in_hyper = \
            self.concat_signature(input_ids=input_ids, signatures=signatures, dom_emb=in_hyper)

        return input_ids, attention_mask, signatures, in_hyper

    def concat_signature(self, input_ids: LongTensor, signatures: LongTensor, dom_emb: FloatTensor) ->\
            Tuple[LongTensor, LongTensor, FloatTensor]:
        """
        gets the input ids of all the DRF signatures and returns their embedding
        Args:
            input_ids: the input ids of the sentences. a tensor of shape (num sentences, seq_length)
            signatures: the ids of the generated DRF signatures.
             a tensor of shape (num sentences, num repetitions, seq_length)
            dom_emb: the embedding of the generated DRF signatures to be inserted to the HN.
             of shape (num sentences, num repetitions, emb_dim)
        Returns:
            the ids of the sentences (shape (num sentences, num repetitions, seq_length)),
             their attention masks (shape (num sentences, num repetitions, seq_length)),
              and the representations of the generated DRF signatures (shape (num sentences, nup repetitions, emb_dim))
        """
        num_sen, num_pre, seq_len = signatures.shape
        signature = signatures.view(-1, seq_len)
        signature = self.tokenizer.batch_decode(signature, skip_special_tokens=True)
        signature = [f'{p}: ' for p in signature]
        signatures = []
        for i in range(0, len(signature), num_pre):
            signatures += signature[i: i + num_pre] + [''] * num_pre
        signature = signatures
        rep_sentences = torch.repeat_interleave(input_ids, 2 * num_pre, dim=0)
        sentences_txt = self.tokenizer.batch_decode(rep_sentences, skip_special_tokens=True)
        sentences = [f'{p}{s}' for p, s in zip(signature, sentences_txt)]
        sentences = self.tokenizer.batch_encode_plus(sentences, max_length=self.h_params.max_seq_length,
                                                     padding="max_length", return_tensors="pt", truncation=True,
                                                     add_special_tokens=True)
        ids, masks = sentences.data["input_ids"], sentences.data["attention_mask"]
        _, seq_len = ids.shape
        ids = ids.view(num_sen, 2 * num_pre, seq_len)
        masks = masks.view(num_sen, 2 * num_pre, seq_len)
        in_hyper = torch.cat([dom_emb, dom_emb], dim=1)
        return ids, masks, in_hyper

    def forward(self, input_ids: LongTensor, signature: FloatTensor = None, attention_mask: LongTensor = None) ->\
            FloatTensor:
        """
        the forward pass of the model
        Args:
            input_ids: the input ids of the batch. a tensor of shape (batch, num repetitions, seq_length)
            attention_mask: the attention masks of the batch to avoid doing attention with pad tokens.
             a tensor of shape (batch, num repetitions, seq_length)
            signature: the embedding of the generated DRF signatures to be inserted to the HN.
             of shape (batch, num repetitions, emb_dim)
        Returns:
            the logits of the label predictions
        """
        batch, num_rep, emb_dim = signature.shape
        input_ids = input_ids.view(batch * num_rep, -1)
        attention_mask = attention_mask.view(batch * num_rep, -1)
        encoder_outputs = self.model.encoder(input_ids, attention_mask=attention_mask)[0]
        signature = signature.view(batch * num_rep, emb_dim)
        logits = self.classify_item(encoder_outputs, signature)
        logits = logits.view(batch, num_rep, -1)
        logits = logits.mean(axis=1)
        return logits

    def test_step(self, batch: Tuple, batch_idx) -> FloatTensor:
        """
        Operates on a single batch of data from the validation set.

        This step is used to generate examples or calculate anything of interest like accuracy.

        Args:
            batch: the output of DataLoader
            batch_idx: the index of this batch.

        Returns:
            A tuple of (loss, generated_texts, labels_texts, sample_ids)
        """
        sentences = batch[0][:, -1, :]
        self.sentences += self.tokenizer.batch_decode(sentences, skip_special_tokens=True)
        return self._eval_step(batch, "test")
