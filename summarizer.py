from transformers import pipeline

class DocumentSummarizer:
    def __init__(self):
        self.model = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

    def hierarchical_summary(self, chunks, final_length=200):
        partials = []

        for c in chunks:
            if len(c.split()) < 60:
                partials.append(c)
            else:
                s = self.model(
                    c,
                    max_length=120,
                    min_length=60,
                    do_sample=False
                )
                partials.append(s[0]["summary_text"])

        combined = " ".join(partials)

        final = self.model(
            combined,
            max_length=final_length,
            min_length=final_length // 2,
            do_sample=False
        )

        return final[0]["summary_text"]
