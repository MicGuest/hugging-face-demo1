from transformers import pipeline
import gradio as gr


model = pipeline(
    "summarization",
)

def predict(prompt):
    #summary = model(prompt)[0]["summary_text"]
    summarizer = pipeline(“summarization”, model=”t5-base”, tokenizer=”t5-base”, framework=”tf”)
    summary=summarizer(context, max_length=130, min_length=60)
    return summary


# create an interface for the model
with gr.Interface(predict, "textbox", "text") as interface:
    interface.launch()
