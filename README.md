# Sentinel AI

GitHub: [suryaguttapalli/Sentinel_AI](https://github.com/suryaguttapalli/Sentinel_AI)

Simple local web app that uses a pretrained **DenseNet-121 CNN** to estimate whether an image or video is AI-generated. It shows a prediction plus AI and real probability scores.

The CNN weights come from [VilaVision/AIgeneratedimagedetection](https://huggingface.co/VilaVision/AIgeneratedimagedetection) and download automatically on first run (~28 MB). The model has three output classes (DALL·E-generated, human-created, and other AI-generated); Sentinel combines the two AI classes into one AI probability.

## Run

Python 3.9+ is required. From this folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501). Upload a `.jpg`, `.png`, `.webp`, `.mp4`, `.mov`, or `.avi` file.

Videos are scored by sampling 8 frames and averaging the AI probability.

First run needs internet so the model can download. After that it uses the local Hugging Face cache. If the download fails, the app shows setup guidance and the underlying error.

## Notes

Scores are probabilistic, not forensic proof. Newer generators, screenshots, and heavy compression can throw the CNN off.
