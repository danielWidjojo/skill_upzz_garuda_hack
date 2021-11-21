

from googleapiclient.discovery import build

def filter_ascii(text):
    text_list = list(text); back_text = ''
    flag = False; ascii_portion = -1; text_len = len(text_list)
    for idx, char in enumerate(text_list):
        if idx == 0:
            char_1 = char
            continue
        else:
            if flag == True: # at the start of ascii integer character
                if char == ';': # end of ascii portion
                    new_char = chr(int(ascii_portion))
                    char_1 = new_char
                    flag = False
                    
                else:
                    ascii_portion += char
                
            elif char == '#' and char_1 == '&': # one before the start of the ascii portion
                flag = True
                ascii_portion = ''
                
            else:
                back_text += char_1
                char_1 = char
                
    back_text += char_1
    return back_text


def translate_language(text, apikey, source = 'en', target = 'id'):
    # Language argument must be encoded according to https://cloud.google.com/translate/docs/languages
    # text: n strings; each string is an essay
    temp = text
    if type(temp) == str:
        temp = [temp]
    
    service = build('translate', 'v2', developerKey=apikey)
    outputs = service.translations().list(source = source, target = target, q = temp).execute()
    
    output_texts = []
    raw_output_texts = outputs['translations']
    
    for raw_text in raw_output_texts:
        output_texts.append(filter_ascii(raw_text['translatedText']))
        
    return output_texts
    

