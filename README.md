# blip-weather-vqa
Visual Question Answering on weather images using BLIP
# Weather VQA with BLIP

**Visual Question Answering on weather images** using Salesforce's BLIP model.

Live Demo: https://huggingface.co/spaces/YOUR_USERNAME/weather-vqa-blip

## Features
- Upload any weather-related image (rain, snow, fog, sunrise, etc.)
- Ask natural questions:
  - "Is it raining?"
  - "Is the sky cloudy?"
  - "Is this daytime or nighttime?"
  - "Can you see the sun?"
  - "What season does this look like?"

## Model
- **Salesforce/blip-vqa-base** (Visual Question Answering)
- Fast inference with cached model

## How to Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/weather-vqa-blip.git
cd weather-vqa-blip
pip install -r requirements.txt
python app.py
