import openai
import os
from dotenv import load_dotenv
import pandas as pd
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load your test set
df = pd.read_csv("/Users/akshayjoshi/Documents/Company_Policies_documents/testset_complete_34.csv")

# Define metrics and prompts
def get_openai_score(metric_name, question, answer, context=None, ground_truth=None):
    base_prompt = f"You are an evaluator scoring the {metric_name} of the following answer."

    prompts = {
        "faithfulness": (
            base_prompt + "\nScore from 1 (hallucinated) to 5 (fully grounded in context).\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
        ),
        "relevancy": (
            base_prompt + "\nScore from 1 (not relevant) to 5 (fully relevant).\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        ),
        "context_recall": (
            base_prompt + "\nScore from 1 (misses key points) to 5 (covers all key info from context).\n"
            f"Context:\n{context}\n\nAnswer:\n{answer}"
        ),
        "correctness": (
            base_prompt + "\nScore from 1 (incorrect) to 5 (fully correct).\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\nGround Truth:\n{ground_truth}"
        ),
        "conciseness": (
            base_prompt + "\nScore from 1 (verbose) to 5 (concise and to-the-point).\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        ),
        "helpfulness": (
            base_prompt + "\nScore from 1 (not helpful) to 5 (very helpful and informative).\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        )
    }

    prompt = prompts.get(metric_name)
    if not prompt:
        raise ValueError(f"Unsupported metric: {metric_name}")

    # Retry if RateLimitError occurs
    for _ in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Only output a single integer score from 1 to 5."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response["choices"][0]["message"]["content"].strip()
        except openai.error.RateLimitError:
            print("⚠️ Rate limit hit. Sleeping for 10s...")
            time.sleep(10)

    return "error"

# Add empty columns to store results
metrics = ["faithfulness", "relevancy", "context_recall", "correctness", "conciseness", "helpfulness"]
for metric in metrics:
    df[metric] = None

print("\n📊 Running Evaluation...\n")
for idx, row in df.iterrows():
    print(f"🔍 Processing Q{idx+1}: {row['question'][:60]}...")
    for metric in metrics:
        score = get_openai_score(
            metric_name=metric,
            question=row["question"],
            answer=row["answer"],
            context=row["context"],
            ground_truth=row["ground_truth"]
        )
        df.at[idx, metric] = score
        print(f"   {metric.title()}: {score}/5")
    print("-" * 40)

# Save results
df.to_csv("internal_document_chatbot/evaluation_results.csv", index=False)
print("\n✅ Evaluation complete! Results saved to evaluation_results.csv")
