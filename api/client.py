import requests
import streamlit as st

def get_llama_response_essay(input_text: str) -> str:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/essay/invoke",
            json={"input": {"topic": input_text}},
            timeout=30  # Optional: prevent hanging forever
        )
        response.raise_for_status()
        return response.json().get("output", "No output returned.")
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"
    except KeyError:
        return "❌ Unexpected response format."

st.set_page_config(page_title="LLAMA2 Essay Generator")

st.title("🦙 LangChain + LLaMA2 Essay Generator")
input_text = st.text_input("Write an essay on:")

if input_text:
    st.info("Generating essay using LLaMA2 via LangServe API...")
    output = get_llama_response_essay(input_text)
    st.markdown("### ✍️ Essay Output")
    st.success(output)
