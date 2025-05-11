from transformers import MarianMTModel, MarianTokenizer

# Supported translation models
MODEL_NAMES = {
    ("sw", "en"): "Helsinki-NLP/opus-mt-sw-en",
    ("en", "sw"): "Helsinki-NLP/opus-mt-en-sw",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es"
}

loaded_models = {}

def load_translation_model(src, tgt):
    key = (src, tgt)
    if key not in loaded_models:
        model_name = MODEL_NAMES[key]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        loaded_models[key] = (tokenizer, model)
    return loaded_models[key]

def translate(text, src, tgt):
    tokenizer, model = load_translation_model(src, tgt)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_to_english(text, lang):
    lang_code = lang.lower()[:2]
    if lang_code in ("es", "sw"):
        return translate(text, lang_code, "en")
    return text

def translate_from_english(text, lang):
    lang_code = lang.lower()[:2]
    if lang_code in ("es", "sw"):
        return translate(text, "en", lang_code)
    return text
