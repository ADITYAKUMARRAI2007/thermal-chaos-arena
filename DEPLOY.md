# Deploying to Hugging Face Spaces

This repo is already Hugging Face Spaces-ready:

- `README.md` contains the Space metadata block
- `app.py` is the Gradio entrypoint
- `requirements.txt` lists runtime dependencies

## Option A: Deploy with Hugging Face CLI

```bash
huggingface-cli login
huggingface-cli repo create thermal-chaos-arena --type space --space_sdk gradio
huggingface-cli upload ADITYAKUMARRAI2007/thermal-chaos-arena . --repo-type space
```

Replace `ADITYAKUMARRAI2007` with your Hugging Face username if different.

## Option B: Deploy from the Hugging Face website

1. Go to https://huggingface.co/new-space
2. Create a Gradio Space named `thermal-chaos-arena`
3. Upload these files/folders:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `thermal_arena/`
   - `scripts/`
