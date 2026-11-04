# VeilBreak demo for a free Hugging Face Space (Docker type, CPU only).
# The app holds no API keys and Llama Guard is NOT in this image: the demo is
# decoders + MiniLM + our classifier, and benchmark numbers are read from results/.
FROM python:3.13-slim

# Hugging Face runs the container as user 1000, so the model cache must be readable by it.
RUN useradd -m -u 1000 app
ENV HOME=/home/app HF_HOME=/home/app/hf
WORKDIR /home/app/veilbreak

# CPU-only torch is far smaller than the default GPU build.
COPY requirements-space.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-space.txt

COPY .streamlit .streamlit
COPY app app
COPY src src
COPY models models
COPY results/RESULTS.md results/catch_rate_by_variant.png results/false_positive_rate_by_variant.png results/
COPY results/smuggling_catch_rate.png results/smuggling_false_positive_rate.png results/
COPY results/classification_metrics.csv results/decoder_metrics.csv results/

# Download MiniLM while building, so the running app never needs the network.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2')" \
    && chown -R app /home/app
ENV HF_HUB_OFFLINE=1

USER app
EXPOSE 7860
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=7860", "--browser.gatherUsageStats=false"]
