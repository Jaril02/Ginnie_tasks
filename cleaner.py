import re
import nltk
import spacy
from config import CLEAN_DIR

nltk.download('punkt')
nlp = spacy.load("en_core_web_trf")

def clean_text(text:str)->str:


    text=text.replace("#","")
    text=text.replace("-","")
    text=text.replace("\n","")
    text=text.replace("/","")
    text=text.replace(r"\s+","",)
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    text=text.strip()
    return text

def split_sen(text:str)->str:
    text=text.replace("•", ". ").replace("-", ". ")
    sentence= nltk.sent_tokenize(text)
    return sentence


def lemmatize(sentence:list)->list:
    lemmatized=[]
    for sent in sentence:
        doc=nlp(sent)
        lemmatized.append(" ".join([token.lemma_ for token in doc]))
    return lemmatized

def pretext(text:str)->list:
    clean=clean_text(text)
    splt=split_sen(clean)
    # lem=lemmatize(splt)
    return splt
