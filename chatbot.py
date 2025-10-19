from openai import OpenAI
from google import genai
from dotenv import load_dotenv
import os


load_dotenv()
client = OpenAI(api_key=os.getenv("GENAI_API_KEY"))

print(client)


def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )
    
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "sbye"]:
            break

        response = chat_with_gpt(user_input)
        print("Chatbot: " , response)


                      
