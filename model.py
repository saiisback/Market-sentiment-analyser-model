from groq import Groq

class Request:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.completion = None  # Store response

    def request(self, question):
        self.completion = self.client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": question}],
            temperature=0.6,
            max_completion_tokens=1024,
            top_p=0.95,
            stream=True,
            reasoning_format="raw"
        )

    def answer(self):
        if self.completion:
            for chunk in self.completion:
                print(chunk.choices[0].delta.content or "", end="")
        else:
            print("No completion available. Call request() first.")
