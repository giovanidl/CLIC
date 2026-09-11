# -*- coding = utf-8 -*-
# @File : forward_adapters.py
"""Adaptadores que sabem montar a chamada model(...) a partir de um batch.

É a única diferença real entre o antigo train.py (embeddings pré-computados)
e train_finetuning.py (tokeniza texto cru pra alimentar o encoder de texto
treinável) — agora é um parâmetro (`forward_fn`) passado pro train_loop /
evaluate_model, em vez de dois arquivos praticamente idênticos.
"""
import os

from transformers import BertTokenizer

# Caminho absoluto compartilhado com Model/model.py (ver comentario la) --
# evita recriar o cache do ClinicalBERT quando o script roda de um cwd
# diferente da raiz do projeto.
bert_pretrain_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model", "BERT_pretrain"
)


def precomputed_forward(model, batch, device):
    """batch = (ecg_emb, text_emb, targets); modelo já recebe tudo pronto."""
    ecg_emb, text_emb, targets = batch
    ecg_emb = ecg_emb.to(device)
    text_emb = text_emb.to(device)
    targets = targets.to(device)

    logits = model(ecg_emb, text_emb)
    return logits, targets


class FinetuningTextForward:
    """batch = (ecg, text_raw, targets); tokeniza o texto ao vivo e chama
    model(ecg, input_ids, attention_mask). Usado no finetuning fim-a-fim
    do encoder de texto (ClinicalBERT).
    """

    def __init__(self, max_length=100):
        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", cache_dir=bert_pretrain_path
        )
        self.max_length = max_length

    def __call__(self, model, batch, device):
        ecg, text, targets = batch
        ecg = ecg.to(device)
        targets = targets.to(device)

        tokens = self.tokenizer(
            list(text),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        logits = model(ecg, input_ids, attention_mask)
        return logits, targets
