from flask import jsonify, request
import requests

def generate_quote(request, client, deployment_name):

    data = request.json
    userPrompt = data.get('prompt')
    systemPrompt = "You are a quote generator. Create an inspirational quote based on the user input."
    
    if userPrompt:

        completion = client.chat.completions.create(
            model=deployment_name,  # e.g. gpt-35-instant
            messages=[
                {
                    "role": "system",
                    "content": systemPrompt,
                },
                {
                    "role": "user",
                    "content": userPrompt,
                },
            ],
        )
        
        print (completion.model_dump_json(indent=2))
        
        quote = completion.choices[0].message.content     

        return jsonify({'quote': quote})

    else:
        return jsonify({'error': 'User prompt parameter is missing.'})

