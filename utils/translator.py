from googletrans import Translator
from langdetect import detect

translator = Translator()

def detect_language(text):
    return detect(text)

def translate_to_english(text):
    return translator.translate(text, dest='en').text

def translate_to_spanish(text):
    return translator.translate(text, dest='es').text
