import os 
import openai 


def gpt3(text): 
    openai.api_key = 'INSERT OPENAI API KEY HERE'
    response = openai.Completion.create(
        engine = 'davinci',
        prompt = text,
        temperature = 0.5, 
        max_tokens = 200, 
        top_p = 1,
        frequency_penalty = 1,
        presence_penalty = 0
    )
    content = response.choices[0].text.split('.')
    return response.choices[0].text