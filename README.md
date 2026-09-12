# Automatic Generation of Dyslexia-Friendly French Educational Texts Using Fine-Tuned LLMs

> **A Phonics-Aware Approach for Moroccan Primary School Children (CP/CE1)**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-lsadouk1111%2Fdyslexia--french--cp--ce1-blue)](https://huggingface.co/datasets/lsadouk1111/dyslexia-french-cp-ce1)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-lsadouk1111%2Fmistral7b--dyslexie--french--cp--ce1-green)](https://huggingface.co/lsadouk1111/mistral7b-dyslexie-french-cp-ce1)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-lecturefacile--dyslexie-orange)](https://huggingface.co/spaces/lsadouk1111/lecturefacile-dyslexie)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

## 📖 Overview

This repository contains the full training and evaluation code for **LectureFacile**, an end-to-end system for automatically simplifying French primary school texts for dyslexic children aged 6–8 (CP and CE1 levels) in Moroccan French-medium schools.

Three fine-tuned models were trained on increasing amounts of data:

| Model | Training pairs | BLEU | SARI |
|-------|---------------|------|------|
| Model 1 | 99 | 61.53 | 79.50 |
| Model 2 | 243 | 68.29 | 83.09 |
| **Model 3 (best)** | **294** | **70.02** | **84.64** |

**Additional evaluation metrics (Model 3):**

| Metric | Score |
|--------|-------|
| SARI Add | 68.18 |
| SARI Keep | 89.48 |
| SARI Delete | 96.25 |
| Flesch Reading Ease (model vs original) | 94.85 vs 92.98 |
| Gunning Fog (model vs original) | 5.95 vs 6.67 |
| ARI (model vs original) | 3.70 vs 4.61 |
| Syllabic segmentation accuracy (LLM-judge) | ~73% |
| Sentence-level hallucination rate | 5.6% |
| Word-level hallucination rate | ~0.7% |

---

## 🌐 Live Application

**LectureFacile** is deployed and publicly accessible at:

👉 **[https://huggingface.co/spaces/lsadouk1111/lecturefacile-dyslexie](https://huggingface.co/spaces/lsadouk1111/lecturefacile-dyslexie)**

100% free — no API key required. Features:
- 📷 Photo upload or webcam capture → automatic OCR → simplified text
- ✏️ Direct text input → simplified text
- Dyslexia-friendly rendering: alternating blue/red syllabic colour coding, Arial font, extended spacing, cream background
- French text-to-speech at 75% speed

---

## 🏗️ System Architecture

```
📷 Photo → Qwen2.5-VL-7B OCR (free, HF ZeroGPU)
         → extracted text
         → Fine-tuned Mistral-7B (free, HF ZeroGPU)
         → Dyslexia-friendly simplified text with syllabic segmentation
```

**Note on inference time:** End-to-end response time (OCR + simplification) ranges from 30–45 seconds for short texts (50–80 words) to 45–60 seconds for longer texts (150–200 words). ZeroGPU reloads model weights per request.

---

## 📁 Repository Structure

```
dyslexia-french-simplification/
│
├── notebooks/
│   ├── dyslexia_train_all_models.ipynb       # Main training notebook (Models 1, 2, 3 + learning curve)
│   ├── dyslexia_readability_analysis.ipynb   # Dataset readability metrics (Flesch, Fog, ARI...)
│   ├── dyslexia_additional_analysis.ipynb    # SARI components, zero-shot comparison,
│   │                                         # inference time, error analysis, Pyphen experiment
│   └── dyslexia_human_evaluation.ipynb       # Human evaluation sample extraction (15 texts)
│
├── hf_space/
│   ├── app.py                                # LectureFacile — Gradio backend (OCR + simplification)
│   └── requirements.txt                      # Space dependencies
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
# Output: La maî-tres-se ex-pli-que u-ne le-çon.
#         El-le par-le aux é-lè-ves de CE1.
```

---

## 📓 Notebooks

### `dyslexia_train_all_models.ipynb`
Main training notebook. Trains 3 fine-tuned versions of Mistral-7B-Instruct-v0.2 with LoRA on increasing training data (99, 243, 294 pairs). Also trains 3 additional learning curve models (50, 150, 200 pairs). Evaluates all models with BLEU and SARI on a fixed test set of 36 pairs.

**Key cells:**
- Cell 0–2: Setup, imports, Drive mount
- Cell 3: Data loading and prompt formatting
- Cell 4: `train_model()` function (LoRA fine-tuning with early stopping)
- Cell 5: `evaluate_on_fixed_test()` function (BLEU + SARI)
- Cell 6–8: Train and evaluate Models 1, 2, 3
- Cell 13: Inference and readability analysis of model outputs
- Cell 14–15: Learning curve models (LC50, LC150, LC200)

### `dyslexia_readability_analysis.ipynb`
Computes 5 automatic readability metrics on training datasets to validate gold-standard simplification quality:
- Flesch Reading Ease (Kandel & Moles French adaptation)
- Gunning Fog Index
- ARI (Automated Readability Index)
- Average sentence length
- Simple word ratio (≤2 syllables)

### `dyslexia_additional_analysis.ipynb`
Additional analyses reported in the paper:
- **Cell 4**: Zero-shot Mistral vs Model 3 qualitative comparison (T174, T316, T054)
- **Cell 5**: SARI component breakdown — Add (68.18) / Keep (89.48) / Delete (96.25)
- **Cell 6**: Inference time measurement on T4 GPU
- **Cell 7**: Error analysis by school level (CP vs CE1)
- **Cell 8–9**: HuggingFace dataset and model card upload
- **Cell 10**: Pyphen post-processing experiment (negative result — typographic vs phonological syllabification)

### `dyslexia_human_evaluation.ipynb`
Extracts 15 model outputs for human evaluation by certified teachers at Groupe Scolaire Aphélie, Berrechid, Morocco.

---

## 📊 Dataset

**[lsadouk1111/dyslexia-french-cp-ce1](https://huggingface.co/datasets/lsadouk1111/dyslexia-french-cp-ce1)**

362 manually annotated parallel text pairs for dyslexia-adapted French text simplification.

| Split | File | Pairs |
|-------|------|-------|
| Train (Model 1) | dataset_m1_train.jsonl | 99 |
| Val (Model 1) | dataset_m1_val.jsonl | 10 |
| Train (Model 2) | dataset_m2_train.jsonl | 243 |
| Val (Model 2) | dataset_m2_val.jsonl | 27 |
| Train (Model 3) | dataset_m3_train.jsonl | 294 |
| Val (Model 3) | dataset_m3_val.jsonl | 32 |
| Test (fixed, all models) | dataset_test_fixed.jsonl | 36 |

**Source distribution:**

| Source | Pairs | % |
|--------|-------|---|
| Nénuphar | 92 | 25.4% |
| Synthetic (CP/CE1) | 100 | 27.6% |
| Étincelles | 60 | 16.6% |
| Kimamila | 49 | 13.5% |
| Coccinelle | 43 | 11.9% |
| Additional authentic texts | 18 | 5.0% |

Each example:
```json
{
  "id": "T174",
  "niveau": "Nen-CP",
  "theme": "Les loups en été",
  "instruction": "Tu es un expert en orthophonie...",
  "input": "Original French CP/CE1 text",
  "output": "Simplified dyslexia-friendly text with syllabic segmentation"
}
```

---

## 🤖 Model

**[lsadouk1111/mistral7b-dyslexie-french-cp-ce1](https://huggingface.co/lsadouk1111/mistral7b-dyslexie-french-cp-ce1)**

| Parameter | Value |
|-----------|-------|
| Base model | Mistral-7B-Instruct-v0.2 (4-bit NF4) |
| Method | LoRA (r=16, alpha=32, dropout=0.05) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 41,943,040 (1.11% of total) |
| Training pairs | 294 |
| Best checkpoint | Step 60 (lowest validation loss) |
| Optimizer | AdamW 8-bit, cosine LR scheduler |
| Hardware | NVIDIA T4 GPU (Google Colab Pro) |
| Framework | Unsloth + TRL |

---

## 🖥️ LectureFacile Space Backend

The `hf_space/` folder contains the full Gradio application code:

- **`app.py`** — Two functions: `extract_text()` (Qwen2.5-VL OCR) and `simplify()` (fine-tuned Mistral-7B). Both decorated with `@spaces.GPU` for ZeroGPU allocation.
- **`requirements.txt`** — Space dependencies

To deploy your own instance:
1. Create a HuggingFace Space (Gradio SDK, ZeroGPU hardware)
2. Upload `app.py` and `requirements.txt`
3. Add your `HF_TOKEN` as a secret in Space settings

---

## 🔬 Key Findings

1. **Fine-tuning is essential** — Zero-shot Mistral produces grammatically incomplete outputs and partial English (BLEU 26.85 vs 70.02 fine-tuned)
2. **Data scaling works** — BLEU and SARI increase monotonically from 50 to 294 training pairs
3. **Sentence segmentation dominates** — Delete (96.25) and Keep (89.48) SARI scores confirm restructuring is the primary learned behaviour
4. **First pipeline-integrable French oral syllabifier** — While desktop tools exist (Coupe-Mots, Syllabes et Compagnie), no Python library or API-integrable system is available. This model is the first trainable, pipeline-integrable French oral syllabifier, achieving ~73% accuracy on unseen texts
5. **BLEU/SARI are insufficient alone** — Model 2 scores higher on BLEU/SARI than Model 1 but worse on readability metrics; both metrics must be reported together

---

## 📝 Citation

```bibtex
@article{sadouk2026dyslexia,
  title={Automatic Generation of Dyslexia-Friendly French Educational Texts
         Using Fine-Tuned LLMs: A Phonics-Aware Approach for Moroccan
         Primary School Children},
  author={Sadouk, Lamyaa and Gadi, Taoufiq},
  journal={},
  year={2026},
  note={Dataset: https://huggingface.co/datasets/lsadouk1111/dyslexia-french-cp-ce1,
        Model: https://huggingface.co/lsadouk1111/mistral7b-dyslexie-french-cp-ce1,
        Code: https://github.com/lsadouk1111/dyslexia-french-simplification,
        App: https://huggingface.co/spaces/lsadouk1111/lecturefacile-dyslexie}
}
```

---

## 👩‍🏫 About

This work was conducted by **Lamyaa Sadouk** (Doctor in Computer Science and AI) and **Taoufiq Gadi**  (Doctor in Computer Science and AI).

The dataset was manually simplified and validated in collaboration with teachers at Groupe Scolaire Aphélie, Berrechid, Morocco, aligned with official French dyslexia accommodation guidelines.

---

## 📄 License

Creative Commons Attribution 4.0 International — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
