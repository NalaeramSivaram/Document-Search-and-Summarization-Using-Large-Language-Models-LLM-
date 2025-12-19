from transformers import pipeline

# Load once (important for speed)
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def llm_answer(context, question):
    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""

    output = qa_pipeline(
        prompt,
        max_length=200,
        do_sample=False
    )

    return output[0]["generated_text"].strip()
