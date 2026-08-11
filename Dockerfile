FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# >>> steam-fetcher — stripped from the public release by build-release.ps1
# fetcher/ is a Node companion that downloads demos from Valve. api.py runs it
# as a subprocess, so Node has to live in this image next to Python.
# Debian's own nodejs package is used rather than an external repo: the base
# image currently provides Node 20, comfortably above the fetcher's minimum
# of 18. Verify with: docker run --rm --entrypoint node <image> --version
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Install Node dependencies before copying the source so this layer stays cached
# across code changes. node_modules is .dockerignore'd, so the Linux build here
# is what ends up in the image rather than anything built on the host.
COPY fetcher/package.json fetcher/package-lock.json ./fetcher/
RUN cd fetcher && npm install --omit=dev --no-audit --no-fund
# <<< steam-fetcher

COPY . .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
