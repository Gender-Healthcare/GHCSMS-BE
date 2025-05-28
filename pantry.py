# === 8. Use OpenAI Chat (Langchain Agent Equivalent) ===
# def summarize_text(text):
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant. Keep the answer short and suitable for Discord."},
#         {"role": "user", "content": f"{text}"}
#     ]
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=300
#     )
#     return response['choices'][0]['message']['content']