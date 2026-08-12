FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the model *and* gate it in the same layer. Running preprocess.py alone
# meant an image could ship a degenerate clustering: the artifact is baked in
# at build time, so if it is never validated here it is never validated at all.
# Chained rather than split into two RUN steps so a failure cannot leave a
# cached layer holding an unvalidated model.
RUN python preprocess.py \
    && python select_k.py --check \
    && python validate_model.py

# Drop privileges. The app only reads its own artifacts, so it has no reason to
# run as root. Done after the build steps, which write into /app.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.solara/cdn \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

# Solara caches proxied CDN assets under /usr/local/share by default, which a
# non-root user cannot write; without this it warns and silently disables the
# asset proxy on every start. (Note its warning text names the wrong variable:
# the setting is Assets.proxy_cache_dir, so the prefix is SOLARA_ASSETS_.)
ENV SOLARA_ASSETS_PROXY_CACHE_DIR=/home/appuser/.solara/cdn

EXPOSE 8765

CMD ["solara", "run", "app.py", "--host=0.0.0.0", "--port=8765"]
