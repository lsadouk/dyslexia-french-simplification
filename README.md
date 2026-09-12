# Automatic Generation of Dyslexia-Friendly French Educational Texts Using Fine-Tuned LLMs

> **A Phonics-Aware Approach for Moroccan Primary School Children (CP/CE1)**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-lsadouk1111%2Fdyslexia--french--cp--ce1-blue)](https://huggingface.co/datasets/lsadouk1111/dyslexia-french-cp-ce1)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-lsadouk1111%2Fmistral7b--dyslexie--french--cp--ce1-green)](https://huggingface.co/lsadouk1111/mistral7b-dyslexie-french-cp-ce1)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-lecturefacile--dyslexie-orange)](https://huggingface.co/spaces/lsadouk1111/lecturefacile-dyslexie)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

## 📖 Overview

This repository contains the full training and evaluation code for **LectureFacile**, an end-to-end system for automatically simplifying French primary school texts for dyslexic children aged 6–8 (CP and CE1 levels) in Moroccan French-medium schools.

**Key results (Model 3 — 294 training pairs):**

| Metric | Score |
|--------|-------|
| BLEU | 70.02 |
| SARI | 84.64 |
| Flesch Reading Ease (model vs original) | 94.85 vs 92.98 |
| Gunning Fog (model vs original) | 5.95 vs 6.67 |
| Syllabic segmentation accuracy | ~73% |

---

## 🏗️ System Architecture

```
📷 Photo → Qwen2.5-VL OCR (free, HF Space)
         → extracted text
         → Fine-tuned Mistral-7B (free, HF Space)
         → Dyslexia-friendly simplified text
```

---

## 📁 Repository Structure

```
dyslexia-french-simplification/
│
├── notebooks/
│   ├── dyslexia_retrain_all_models.ipynb       # Main training notebook (Models 1, 2, 3)
│   ├── dyslexia_readability_analysis.ipynb     # Dataset readability metrics
│   ├── dyslexia_additional_analysis.ipynb      # SARI components, inference time,
│   │                                           # zero-shot comparison, error analysis
│   └── dyslexia_human_evaluation.ipynb         # Human evaluation form extraction
│
├── app/
│   └── LectureFacile.html                      # Tablet application (single HTML file)
│
├── hf_space/
│   ├── app.py                                  # HuggingFace Space — Gradio backend
│   └── requirements.txt                        # Space dependencies
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install unsloth transformers peft bitsandbytes accelerate datasets evaluate sacrebleu
pip install git+https://github.com/feralvam/easse.git
```

### 2. Run inference with the fine-tuned model

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lsadouk1111/mistral7b-dyslexie-french-cp-ce1",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.for_inference(model)
tokenizer.pad_token = tokenizer.eos_token

INSTRUCTION = (
    "Tu es un expert en orthophonie et en éducation inclusive. "
    "Simplifie le texte suivant pour un enfant dyslexique francophone âgé de 6 à 8 ans. "
    "Applique ces règles obligatoires : "
    "(1) Phrases courtes de 8 à 10 mots maximum. "
    "(2) Structure Sujet + Verbe + Complément uniquement. "
    "(3) Vocabulaire simple — remplace les mots de plus de 3 syllabes. "
    "(4) Découpage syllabique avec tirets : ma-man, jar-din, é-lè-ve. "
    "(5) Conserve TOUS les détails narratifs. "
    "(6) Retour à la ligne à chaque phrase."
)

def simplify(text):
    prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{text}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=600, temperature=0.3,
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    if "### " in result:
        result = result[:result.index("### ")].strip()
    return result

text = "La maîtresse explique une leçon difficile aux élèves de CE1."
print(simplify(text))
```

---

## 📓 Notebooks

### `dyslexia_retrain_all_models.ipynb`
Main training notebook. Trains 3 fine-tuned versions of Mistral-7B-Instruct-v0.2 with LoRA on increasing amounts of training data (99, 243, 294 pairs). Evaluates each model with BLEU and SARI on a fixed test set of 36 pairs.

**Key cells:**
- Cell 0–2: Setup and imports
- Cell 3: Data loading and prompt formatting
- Cell 4: `train_model()` function (LoRA fine-tuning)
- Cell 5: `evaluate_on_fixed_test()` function (BLEU + SARI)
- Cell 6–8: Train and evaluate Models 1, 2, 3
- Cell 13: Inference and readability analysis of model outputs
- Cell 14–15: Learning curve (LC50, LC150, LC200)

### `dyslexia_readability_analysis.ipynb`
Computes 5 automatic readability metrics (Flesch, Gunning Fog, ARI, avg sentence length, simple word ratio) on the training datasets to validate the quality of the gold-standard simplifications.

### `dyslexia_additional_analysis.ipynb`
Additional analyses reported in the paper:
- **Cell 4**: Zero-shot vs Model 3 qualitative comparison (T174, T316, T054)
- **Cell 5**: SARI component breakdown (Add/Keep/Delete)
- **Cell 6**: Inference time measurement
- **Cell 7**: Error analysis by school level (CP vs CE1)
- **Cell 8–9**: HuggingFace dataset and model card upload
- **Cell 10**: Pyphen post-processing experiment (negative result)

### `dyslexia_human_evaluation.ipynb`
Extracts 15 model outputs for human evaluation by teachers at Groupe Scolaire Aphélie.

---

## 📊 Dataset

The dataset is publicly available on HuggingFace:
**[lsadouk1111/dyslexia-french-cp-ce1](https://huggingface.co/datasets/lsadouk1111/dyslexia-french-cp-ce1)**

| Split | File | Pairs |
|-------|------|-------|
| Train (Model 1) | dataset_m1_train.jsonl | 99 |
| Val (Model 1) | dataset_m1_val.jsonl | 10 |
| Train (Model 2) | dataset_m2_train.jsonl | 243 |
| Val (Model 2) | dataset_m2_val.jsonl | 27 |
| Train (Model 3) | dataset_m3_train.jsonl | 294 |
| Val (Model 3) | dataset_m3_val.jsonl | 32 |
| Test (fixed) | dataset_test_fixed.jsonl | 36 |

Each example contains:
```json
{
  "id": "T174",
  "niveau": "Nen-CP",
  "theme": "Les loups en été",
  "instruction": "Tu es un expert en orthophonie...",
  "input": "Original French text",
  "output": "Simplified dyslexia-friendly text"
}
```

---

## 🤖 Model

Fine-tuned model: **[lsadouk1111/mistral7b-dyslexie-french-cp-ce1](https://huggingface.co/lsadouk1111/mistral7b-dyslexie-french-cp-ce1)**

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base model | Mistral-7B-Instruct-v0.2 (4-bit) |
| Method | LoRA (r=16, alpha=32) |
| Training pairs | 294 |
| Best checkpoint | Step 60 |
| Hardware | NVIDIA T4 GPU (Google Colab Pro) |

---

## 🌐 LectureFacile App

The tablet application is available at:
**[https://huggingface.co/spaces/lsadouk1111/lecturefacile-dyslexie](https://huggingface.co/spaces/lsadouk1111/lecturefacile-dyslexie)**

The standalone HTML file (`app/LectureFacile.html`) can also be deployed on GitHub Pages or run locally. It requires no installation.

**Pipeline:**
```
📷 Photo → Qwen2.5-VL OCR (HF Space, free)
         → Fine-tuned Mistral-7B (HF Space, free)
         → Dyslexia-friendly text with syllabic segmentation
```

---

## 📝 Citation

```bibtex
@article{sadouk2026dyslexia,
  title={Automatic Generation of Dyslexia-Friendly French Educational Texts
         Using Fine-Tuned LLMs: A Phonics-Aware Approach for Moroccan
         Primary School Children},
  author={Sadouk, Lamyaa},
  journal={},
  year={2026},
  note={Dataset: https://huggingface.co/datasets/lsadouk1111/dyslexia-french-cp-ce1,
        Model: https://huggingface.co/lsadouk1111/mistral7b-dyslexie-french-cp-ce1,
        Code: https://github.com/lsadouk1111/dyslexia-french-simplification}
}
```

---

## 👩‍🏫 About

This work was conducted by **Lamyaa Sadouk**, Doctor in Computer Science and AI, certified primary school teacher at **Groupe Scolaire Aphélie, Berrechid, Morocco**.

The dataset was manually simplified and validated in collaboration with teachers at Aphélie school, aligned with official French dyslexia accommodation guidelines.

---

## 📄 License

Creative Commons Attribution 4.0 International — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
