from google import genai
from dotenv import load_dotenv
import os
import pandas
import numpy
import settings

load_dotenv()
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
instructions  = open("instructions.txt")
instructions = instructions.read()

pd = pandas.read_csv('salary_data.csv')
pd = pd.to_numpy()
pd = str(pd)

response = client.models.generate_content(
            model= "gemini-2.0-flash",
            contents = instructions
            )

def run_chatbot():
        while True:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit", "sbye"]:
                    break

                response = client.models.generate_content(
                    model= "gemini-2.0-flash",
                    contents = user_input + instructions
                    )
                
                print("Chatbot: ", response.text)


def chat(user_input):
        while True:
                user_input = user_input
                if user_input.lower() in ["quit", "exit", "sbye"]:
                    break

                response = client.models.generate_content(
                    model= "gemini-2.0-flash",
                    contents = user_input + instructions
                    )
                
                print("Chatbot: ", response.text)
                return response.text
