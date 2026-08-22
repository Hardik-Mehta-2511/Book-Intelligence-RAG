from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5-mini",
    input="Say hello in one sentence."
)

print(response.output_text)