from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from mysite.spell_checker import spell_checker
from mysite.thesaurus import find_synonym
from mysite.gcp_translate import translate_language
from mysite.gcp_ocr import vision_read_image
from mysite.BERT.bert_predict_temp import predict as predict_bert, load_model
from mysite.gpt_essay_prompter import gpt3
import json

model = load_model(state_dict_path='static/bert/rater1_trait2_BERT_statedict')
model1 = load_model(state_dict_path='static/bert/rater1_trait3_BERT_statedict')
model2 = load_model(state_dict_path='static/bert/rater1_trait4_BERT_statedict')
print("initialize")

def index(request):
    
    global model
    
    if (request.method == "POST"):
        
        print("POST METHOD")
        if 'image_upload' in request.POST:
            
            
            APIKEY = "INSERT GCP API KEY HERE"
            inImg = request.FILES['img_essay'].read()
            
            
            text = vision_read_image(inImg, APIKEY)        
            essay, table_correction = spell_checker(text)
            y_pred,_ =predict_bert(model, data_txt= text)
            y_pred1,_ =predict_bert(model1, data_txt= text)
            y_pred2,_ =predict_bert(model2, data_txt= text)
            grade_string = f" Organization : {y_pred[0]} / 3, Style : {y_pred1[0]} / 3, and Convention : {y_pred2[0]} / 3"
            dict1 = [{"text" :essay,
                "table" :table_correction,
                 "grade": grade_string}]
            context = {"analysis" : dict1}
            return render(request, 'result.html',context)

        # get essay text
        text = request.POST.get("essay")
        essay, table_correction = spell_checker(text)
        y_pred,_ =predict_bert(model, data_txt= text)
        y_pred1,_ =predict_bert(model1, data_txt= text)
        y_pred2,_ =predict_bert(model2, data_txt= text)
        #2 organization, 3 style, 4 convention
        grade_string = f" Organization : {y_pred[0]} / 3, Style : {y_pred1[0]} / 3, and Convention : {y_pred2[0]} / 3"
        dict1 = [{"text" :essay,
                "table" :table_correction,
                 "grade": grade_string}]
        context = {"analysis" : dict1}
        return render(request, 'result.html',context)
        # return HttpResponse("Congrats " + str(essay))
    return render(request, 'index.html')

def result(request):
    return render(request, 'result.html')

def translate(request):
    print ('Translate request')
    APIKEY = "INSERT GCP API KEY HERE"

    request = str(request.body)
    dict_responce = json.loads(request[2:-1])
    essay = dict_responce['text_to_translate']
    lang_target = dict_responce['lang_to_translate']
    if lang_target == "id":
        lang_source = "en"
    else:
        lang_source = "id"
    output_texts = translate_language(essay, APIKEY,lang_source,lang_target)
    result = ""
    for output in output_texts:
        result += output
    return HttpResponse(result)

def thesaurus(request):
    print ('thesaurus request')
    essay = str(request.body)
    essay = essay[2:-1]
    
    return HttpResponse(find_synonym(essay))

def prompter(request):
    print ('prompter request')
    essay = str(request.body)
    essay = essay[2:-1]
    return HttpResponse(gpt3(essay))