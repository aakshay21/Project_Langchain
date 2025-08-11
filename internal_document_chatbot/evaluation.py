import openai
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


df = pd.read_csv("/Users/akshayjoshi/Documents/Company_Policies_documents/testset.csv")

def get_openai_score(metric_name, question, answer,context = None, ground_truth = None):
    base_prompt = f"You are an evaluator scoring the {metric_name} of the following answer."

    #Custom Prompts per metric

    prompts = {
        "faithfulness":(
            base_prompt + "\nScore from 1 (hallucinated) to 5 (fully grounded in context), \n"
            f"Context:\n{context}\n\nQuestion: \n{question}\n\nAnswer:\n{answer}"
        ),
            "relevancy": (
            base_prompt + "\nScore from 1 (not relevant) to 5 (fully relevant).\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        ),
        "context_recall": (
            base_prompt + "\nScore from 1 (misses most points) to 5 (covers all key info from context).\n"
            f"Context:\n{context}\n\nAnswer:\n{answer}"
        ),
        "correctness": (
            base_prompt + "\nScore from 1 (incorrect) to 5 (fully correct).\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\nGround Truth:\n{ground_truth}"
        ),
        "conciseness": (
            base_prompt + "\nScore from 1 (verbose/off-topic) to 5 (concise and to-the-point).\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        ),
        "helpfulness": (
            base_prompt + "\nScore from 1 (not helpful) to 5 (very helpful and informative).\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        )
    }


    ##Fallback
    prompt = prompts.get(metric_name)
    if not prompt:
        raise ValueError(f"Unsupported metric: {metric_name}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or use gpt-4 if needed
        messages=[
            {"role": "system", "content": "Only output a single integer score from 1 to 5."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["choices"][0]["message"]["content"].strip()

# Example usage
question = "What is the maternity leave duration?"
answer = "The company offers 26 weeks of maternity leave."
context = "Eligible employees can avail 26 weeks of maternity leave as per HR policy."
ground_truth = "Maternity leave is granted for 26 weeks."

metrics_to_evaluate = ["faithfulness", "relevancy", "context_recall", "correctness", "conciseness", "helpfulness"]

print("\n📊 Evaluation Results:\n")
for metric in metrics_to_evaluate:
    score = get_openai_score(metric, question, answer, context=context, ground_truth=ground_truth)
    print(f"{metric.title()}: {score}/5")