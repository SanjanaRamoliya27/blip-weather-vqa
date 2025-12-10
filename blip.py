
import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load model 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "blip_model"

def load_blip():
    if not os.path.exists(MODEL_DIR):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        processor.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
    else:
        processor = BlipProcessor.from_pretrained(MODEL_DIR)
        model = BlipForQuestionAnswering.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return model, processor

model, processor = load_blip()

def vqa(image, question):
    if image is None:
        return "❌ No image uploaded. Please try dragging a file or clicking upload."
    if not question:
        return "❌ No question. Ask something like 'Is it raining?'"
    
    try:
        inputs = processor(image.convert("RGB"), question, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30)
        answer = processor.decode(output[0], skip_special_tokens=True)
        return f"✅ Answer: {answer.strip().capitalize()}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Simple Gradio Interface (Test Upload Here)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Test Image Upload")
    gr.Markdown("Drag an image here or click to upload. If this works, your issue is fixed!")
    
    img = gr.Image(type="pil", label="Upload Test Image", height=300)  # Explicit height for visibility
    question = gr.Textbox(label="Question", placeholder="e.g., What do you see?")
    btn = gr.Button("Submit")
    output = gr.Textbox(label="Result")
    
    btn.click(vqa, inputs=[img, question], outputs=output)

demo.launch()
