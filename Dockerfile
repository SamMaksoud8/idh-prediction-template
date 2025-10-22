FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash tini build-essential \
    && rm -rf /var/lib/apt/lists/*

# user
ARG APP_USER=appuser
RUN useradd -m -u 1000 ${APP_USER}
ENV HOME=/home/${APP_USER}
WORKDIR /app

# === Install FRPC for Gradio share ===
# If you're building on arm64, change the URL to .../frpc_linux_arm64
#RUN mkdir -p "${HOME}/.cache/huggingface/gradio/frpc" \
#    && curl -fsSL "https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64" \
#    -o "${HOME}/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3" \
#    && chmod +x "${HOME}/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3" \
#    && chown -R ${APP_USER}:${APP_USER} "${HOME}/.cache"
# === End FRPC install ===

# Copy code
COPY . /app

# Dependencies (avoid installing extras like ".[all]" unless you really need them)
RUN if [ -f requirements.txt ]; then \
    pip install --no-cache-dir -r requirements.txt; \
    elif [ -f pyproject.toml ]; then \
    pip install --no-cache-dir .; \
    else \
    echo "No dependency manifest found; continuing"; \
    fi

ENV PYTHONPATH=/app/src PORT=8080
USER ${APP_USER}
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "src/idh/app/csv_prediction.py"]