
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

def find_synonym (target_word):
    
    synonyms = []
    word = target_word
    for ss in wordnet.synsets(word):
        print(ss.name(), ss.lemma_names())
        for indiv_synonym in ss.lemma_names():
            if indiv_synonym not in synonyms and indiv_synonym != word:
                synonyms.append(indiv_synonym)    
    
    list_str = '<ul class="list-group list-group-flush">'
    for word in synonyms[:10]:
        list_str += f'<li class="list-group-item">{word}</li>'
        
    list_str += "</ul>"
    return list_str