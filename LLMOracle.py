import litellm
from litellm import completion, exceptions
def get_llm_response(query):
    try:
        response = completion(
            model="ollama/llama3",
            messages=[{"content": query, "role": "user"}],
            api_base="https://ollama.vasilis.pw"
        )
        return response
    except exceptions.APIConnectionError as e:
        print(f"API Connection Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None
