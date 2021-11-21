from spellchecker import SpellChecker
import pandas as pd
import numpy as np

spell = SpellChecker(language = 'en', case_sensitive= True)

def spell_checker(essay):
    correction_count = {}
    essay_list = essay.replace('.', '').replace(',', '').replace('!', '').split()
    correction_list = []

    for word in essay_list:
        misspell = spell.unknown([word])

        if len(misspell) == 0:
            correction_list.append((False, word))

        else:
            if correction_count.get(word) == None:
                correction_count[word] = 1
            else:
                correction_count[word] += 1

            for err_word in misspell:
                correction = spell.correction(err_word)

                correction_list.append((True, correction))
        
    appended_string = ""
    span_beginning = '<span style="color:red;"> '
    span_end = '</span> '
    arr_wrong_and_corrected_word  = []
    for index, processed in enumerate(correction_list):
        status, corrected_string = processed 
        if status == False :
            appended_string += essay_list[index] + ' '
        
        else :
            arr_wrong_and_corrected_word.append([essay_list[index], corrected_string])
            appended_string += span_beginning + essay_list[index] + ' ' + span_end
    table_string = '<table class="table table-striped"> <thead> <tr> <th scope="col">index</th> <th scope="col">Wrong Spelling</th> <th scope="col">Corrected Spelling</th> </tr> </thead><tbody>'
    for index,tuples  in enumerate(arr_wrong_and_corrected_word) :
        wrong_word, corrected_word = tuples
        table_string += f'<th scope="row">{index + 1}</th><td>{wrong_word}</td> <td>{corrected_word}</td> </tr>'

    table_string += '</tbody></table>'
    return appended_string, table_string