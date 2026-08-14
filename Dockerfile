FROM python:3.11-slim

WORKDIR /app

# Created before anything is copied so COPY can set ownership directly. A
# `chown -R` afterwards would rewrite every file into a second layer, which
# costs roughly the size of the application again.
RUN useradd --create-home --uid 1000 pr1maly

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
COPY --chown=pr1maly:pr1maly fetcher/package.json fetcher/package-lock.json ./fetcher/
RUN cd fetcher && npm install --omit=dev --no-audit --no-fund \
    && npm cache clean --force \
    && chown -R pr1maly:pr1maly /app/fetcher

COPY --chown=pr1maly:pr1maly . .

RUN mkdir -p /app/data && chown pr1maly:pr1maly /app/data

# The application runs as uid 1000, not root — but PID 1 starts as root so it
# can hand over an existing bind-mounted ./data first. Upgrading from an image
# that ran as root leaves the database owned by root, and uid 1000 can read it
# and not write it; SQLite calls that "attempt to write a readonly database"
# and names nothing that would lead you to ownership.
#
# The entrypoint fixes what is wrong, then drops privileges with setpriv and
# execs, so the app itself never runs as root. Pass `user:` in compose to skip
# even that brief root step — the script detects it and does nothing:
#     docker compose run --user "$(id -u):$(id -g)" api
#
# The sed is not superstition: the build context is the working tree, and on
# Windows that tree has CRLF endings, which make a shell script unrunnable.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

# Checked with Python, which is already in the image — installing curl for this
# would add its own dependency tree for one HTTP request.
#
# /api/health is deliberately cheap and does not touch the database: a check
# every thirty seconds would otherwise keep opening connections, and a long
# import would make the container report unhealthy while working perfectly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
