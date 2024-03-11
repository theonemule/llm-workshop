from flask import Flask, request, jsonify, send_from_directory
import whisper
from io import BytesIO
import os
import openai

openai.api_key="56bf5150b5584c7983e8af95dc2216c6"
openai.api_base="https://ai-dictate.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type='azure'
openai.api_version = '2023-05-15' # this might change in the future

deployment_name='ai-dictate' #This will correspond to the custom name you chose for your deployment when you deployed a model. 

app = Flask(__name__, static_folder='static')
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file temporarily
        filename = 'temp_audio.ogg'
        file.save(filename)

        # Transcribe the audio file
        result = model.transcribe(filename)

        # Remove the temporary file
        os.remove(filename)
        
        print(result["text"])



        response = openai.ChatCompletion.create(
            engine=deployment_name, # engine = "deployment_name".
            messages=[
                {"role": "system", "content": """You are an assistant working with transcribed text from a novelist speaking to a computer. Take the transcribed text and change the tone to a novel. Do not add to the text. Try to preserve the text as much as possible. Use the following rules to guide the style:
Tone:
Dramatic: The passage is filled with dramatic events, from the initial disappointment of finding the well dry to the life-and-death struggle in the hospital. The language and descriptions create a sense of tension and suspense.
Straightforward: Do not use a poetic style with its use of vivid imagery and metaphors. 
Urgent: The urgency of the situation is conveyed through phrases like "frantic journey," "treacherous obstacles," and "race against time." The passage emphasizes the need for quick action and the gravity of the circumstances.
Style:
Descriptive: The passage is highly descriptive, painting a detailed picture of the events as they unfold. It uses sensory language to help readers visualize the scenes and feel the emotions of the characters.
Narrative: The style is narrative in nature, as it tells a story with a clear sequence of events and a central conflict. It follows the journey of Jack and Jill, from their quest for water to the life-threatening situation and their ultimate survival.
Emotional: The passage evokes strong emotions, especially in moments of crisis and determination. It explores themes of resilience and the will to survive in the face of adversity."""},
                {"role": "user", "content": result["text"]}
            ]
        )        
        
        print(response)
        # print(response['choices'][0]['message']['content'])

        


        return jsonify({'transcription': result["text"],'formatted': response['choices'][0]['message']['content']})

    return jsonify({'error': 'Invalid file'}), 400

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
# 172.24.89.138