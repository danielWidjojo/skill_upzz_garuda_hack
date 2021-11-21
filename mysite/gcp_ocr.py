import os
import base64
from googleapiclient.discovery import build
def encode_image_vision_api(image_content):
    
    return base64.b64encode(image_content).decode('utf-8')
    
def vision_read_image(image, apikey):
    vservice = build('vision', 'v1', developerKey=apikey)
    request = vservice.images().annotate(body={
        'requests': [{
                'image': {
                    'content': encode_image_vision_api(image)
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 3,
                }]
            }],
        })
    responses = request.execute(num_retries=3)
    text = responses['responses'][0]['fullTextAnnotation']['text']
    
    return text


    

