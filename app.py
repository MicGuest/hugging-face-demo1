from transformers import pipeline
import gradio as gr


model = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

def predict(prompt):
    summary = model(context, max_length=130, min_length=60)
    return summary


# create an interface for the model
with gr.Interface(predict, "textbox", "text") as interface:
    interface.launch()
