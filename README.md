---
title: Named Entity Recognition
emoji: 🏷️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
---

# Named Entity Recognition (NER)

A Transformer encoder model built from scratch using PyTorch, trained on the CoNLL-2003 dataset.

## Architecture
- **Type:** Transformer Encoder + Classification Head
- **Encoder Layers:** 3
- **Attention Heads:** 8
- **Model Dimension:** 256
- **Feed-Forward Dimension:** 1024
- **Parameters:** ~8.4M
- **Dataset:** CoNLL-2003 (~14K sentences)

## Entity Types
- **PER** — Person names
- **ORG** — Organizations
- **LOC** — Locations
- **MISC** — Miscellaneous entities

## How It Works
Every component — embeddings, positional encoding, multi-head attention, feed-forward layers — was implemented from scratch. No pre-trained weights or HuggingFace model classes were used. Only the encoder part of the Transformer is used since NER is a token classification task.
