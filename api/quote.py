from flask import jsonify, request
import requests


def generate_quote(request, client, deployment_name):
    data = request.json
    userPrompt = data.get("prompt")
    systemPrompt = "As a quote generator, your task is to craft an inspirational and philosophical quote tailored to software developers. The quote should be based on a theme or topic provided by the user, with a focus on fostering a motivation to learn and a joy of discovery. The tone should be uplifting, potentially incorporating poetic elements for added inspiration."

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

        print(completion.model_dump_json(indent=2))

        quote = completion.choices[0].message.content

        return jsonify({"quote": quote})

    else:
        return jsonify({"error": "User prompt parameter is missing."})
