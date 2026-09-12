import gradio as gr
import spaces
import torch
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

HF_TOKEN = os.environ.get("HF_TOKEN", None)
SIMPLIF_MODEL_ID = "lsadouk1111/mistral7b-dyslexie-french-cp-ce1"
SIMPLIF_BASE     = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
OCR_MODEL_ID     = "Qwen/Qwen2.5-VL-7B-Instruct"

_simplif_model     = None
_simplif_tokenizer = None
_ocr_model         = None
_ocr_proc          = None

INSTRUCTION = (
    "Tu es un expert en orthophonie et en éducation inclusive. "
    "Simplifie le texte suivant pour un enfant dyslexique francophone age de 6 a 8 ans. "
    "Applique ces regles obligatoires : "
    "(1) Phrases courtes de 8 a 10 mots maximum. "
    "(2) Structure Sujet + Verbe + Complement uniquement. "
    "(3) Vocabulaire simple - remplace les mots de plus de 3 syllabes. "
    "(4) Decoupage syllabique avec tirets : ma-man, jar-din, e-le-ve. "
    "(5) Conserve TOUS les details narratifs. "
    "(6) Retour a la ligne apres chaque phrase."
)

# ── OCR ────────────────────────────────────────────────────────────
@spaces.GPU
def extract_text(image):
    global _ocr_model, _ocr_proc
    if image is None:
        return "Veuillez uploader une image."
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    if _ocr_model is None:
        print("Chargement du modele OCR...")
        _ocr_proc = AutoProcessor.from_pretrained(
            OCR_MODEL_ID, trust_remote_code=True, token=HF_TOKEN
        )
        _ocr_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            OCR_MODEL_ID, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="cuda", token=HF_TOKEN,
        )
        _ocr_model.eval()
        print("OCR pret.")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extrais uniquement le texte principal de cette image de manuel scolaire francais. Retourne SEULEMENT le texte brut sans commentaire."}
        ]
    }]
    text_prompt = _ocr_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _ocr_proc(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated_ids = _ocr_model.generate(**inputs, max_new_tokens=1000)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return _ocr_proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

# ── Simplification ─────────────────────────────────────────────────
@spaces.GPU
def simplify(text):
    global _simplif_model, _simplif_tokenizer
    if not text.strip():
        return ""
    if _simplif_model is None:
        print("Chargement du modele de simplification...")
        _simplif_tokenizer = AutoTokenizer.from_pretrained(SIMPLIF_MODEL_ID, token=HF_TOKEN)
        _simplif_tokenizer.pad_token = _simplif_tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            SIMPLIF_BASE, quantization_config=bnb_config,
            device_map="cuda", token=HF_TOKEN,
        )
        _simplif_model = PeftModel.from_pretrained(base_model, SIMPLIF_MODEL_ID, token=HF_TOKEN)
        _simplif_model.eval()
        print("Modele de simplification pret.")
    prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{text}\n\n### Response:\n"
    inputs = _simplif_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        outputs = _simplif_model.generate(
            **inputs, max_new_tokens=600, temperature=0.3,
            do_sample=True, top_p=0.9, repetition_penalty=1.1,
            eos_token_id=_simplif_tokenizer.eos_token_id,
            pad_token_id=_simplif_tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = _simplif_tokenizer.decode(generated, skip_special_tokens=True)
    if "### " in result:
        result = result[:result.index("### ")].strip()
    return result

# ── Render dyslexia-friendly HTML ──────────────────────────────────
def render_dyslexia(text):
    if not text.strip():
        return "<div style='color:#CBD5E0;text-align:center;padding:40px'>Le texte simplifié apparaîtra ici...</div>"
    colors = ['#2B6CB0', '#C53030']
    lines  = text.split('\n')
    syl_idx = 0
    html = '''<div style="
        font-family: Arial, Verdana, sans-serif;
        font-size: 22px;
        line-height: 2.2;
        letter-spacing: 0.12em;
        word-spacing: 0.25em;
        text-align: left;
        background: #FFFDF0;
        border: 2px solid #F6E05E;
        border-radius: 14px;
        padding: 28px 24px;
        min-height: 120px;
    ">'''
    for line in lines:
        if not line.strip():
            html += '<br>'
            continue
        html += '<p style="margin-bottom:10px">'
        words = line.split(' ')
        for wi, word in enumerate(words):
            if not word:
                continue
            parts = word.split('-')
            for si, syl in enumerate(parts):
                if not syl:
                    continue
                color = colors[syl_idx % 2]
                html += f'<span style="color:{color};font-weight:700">{syl}</span>'
                if si < len(parts) - 1:
                    html += '<span style="color:#CBD5E0">-</span>'
                syl_idx += 1
            if wi < len(words) - 1:
                html += ' '
        html += '</p>'
    html += '</div>'
    return html

# ── Combined functions ─────────────────────────────────────────────
def simplify_and_render(text):
    result = simplify(text)
    return result, render_dyslexia(result)

def ocr_then_simplify(image):
    text = extract_text(image)
    if not text or text.startswith("Veuillez"):
        return text, "", render_dyslexia("")
    result = simplify(text)
    return text, result, render_dyslexia(result)

# ── Gradio UI ──────────────────────────────────────────────────────
css = """
.result-box { background: #FFFDF0 !important; }
.header { text-align:center; padding:16px 0; }
"""

with gr.Blocks(title="LectureFacile", theme=gr.themes.Soft(), css=css) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:20px 0 10px">
      <h1 style="color:#2B6CB0;font-size:2em;margin-bottom:6px">📖 LectureFacile</h1>
      <p style="color:#718096;font-size:1.05em">Simplification automatique de textes pour enfants dyslexiques (CP/CE1)</p>
      <span style="background:#EBF4FF;color:#2B6CB0;border-radius:20px;padding:4px 16px;font-size:0.9em;font-weight:bold">
        ✅ 100% gratuit — aucune clé API requise
      </span>
    </div>
    """)

    with gr.Tabs():

        # Tab 1 — Photo
        with gr.TabItem("📷 Photo du manuel"):
            gr.Markdown("### Photographiez ou importez une page de manuel scolaire")
            gr.Markdown("⏱️ *La première utilisation charge les modèles — prévoir environ 60 secondes.*")
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Photo du manuel",
                        sources=["webcam", "upload"],
                    )
                    btn_auto = gr.Button(
                        "📷 Lire et simplifier automatiquement",
                        variant="primary", size="lg"
                    )
                with gr.Column(scale=1):
                    ocr_output = gr.Textbox(
                        label="📝 Texte extrait (modifiable)",
                        lines=4,
                        placeholder="Le texte extrait apparaîtra ici..."
                    )
                    btn_simplify_ocr = gr.Button("✨ Simplifier ce texte", variant="secondary")

            simplified_raw_1 = gr.Textbox(visible=False)
            result_photo = gr.HTML(render_dyslexia(""))
            gr.Markdown("*Police Arial · Espacement adapté · Couleurs alternées par syllabe*")

            btn_auto.click(
                fn=ocr_then_simplify,
                inputs=image_input,
                outputs=[ocr_output, simplified_raw_1, result_photo]
            )
            btn_simplify_ocr.click(
                fn=simplify_and_render,
                inputs=ocr_output,
                outputs=[simplified_raw_1, result_photo]
            )

        # Tab 2 — Texte direct
        with gr.TabItem("✏️ Écrire ou coller un texte"):
            gr.Markdown("### Tapez ou collez directement un texte à simplifier")
            gr.Markdown("⏱️ *La première utilisation charge le modèle — prévoir environ 60 secondes.*")
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="📄 Texte original",
                        lines=8,
                        placeholder="Ex: La maîtresse explique une leçon difficile aux élèves de CE1.",
                    )
                    btn_simplify = gr.Button("✨ Simplifier ce texte", variant="primary", size="lg")
                with gr.Column(scale=1):
                    simplified_raw_2 = gr.Textbox(visible=False)
                    result_text = gr.HTML(render_dyslexia(""))

            gr.Markdown("*Police Arial · Espacement adapté · Couleurs alternées par syllabe*")

            btn_simplify.click(
                fn=simplify_and_render,
                inputs=text_input,
                outputs=[simplified_raw_2, result_text]
            )

            gr.Examples(
                examples=[
                    ["Un chat est sur le toit. Il regarde les oiseaux. Un oiseau s'envole."],
                    ["La cloche de l'école sonne trois fois par jour. Elle sonne le matin pour l'entrée."],
                    ["C'est le soir, Lila a rendez-vous avec ses amis imaginaires. Elle voit un monstre qui s'appelle Doli."],
                ],
                inputs=text_input,
                label="Exemples de textes CP/CE1"
            )

    gr.HTML("""
    <div style="text-align:center;padding:16px;color:#718096;font-size:0.85em;border-top:1px solid #E2E8F0;margin-top:20px">
      🔬 Modèle fine-tuné sur 294 paires CP/CE1 &nbsp;|&nbsp;
      📊 BLEU 70.02 · SARI 84.64 &nbsp;|&nbsp;
      📄 <a href="https://github.com/lsadouk1111/dyslexia-french-simplification" target="_blank">GitHub</a> &nbsp;|&nbsp;
      🤗 <a href="https://huggingface.co/datasets/lsadouk1111/dyslexia-french-cp-ce1" target="_blank">Dataset</a>
    </div>
    """)

demo.launch()
