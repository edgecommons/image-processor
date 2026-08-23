# Container image for the ImageProcessor edgecommons component.
#
# The image is CPU-only: it installs the CPU `onnxruntime` wheel from requirements.txt, which is
# what the parity suite runs on (D-IP-14). A GPU image installs the `gpu` extra
# (`onnxruntime-gpu`) on a CUDA base image instead, and is a later phase of this component.
#
# Build from THIS directory and load or push it, then set `image:` in k8s/deployment.yaml:
#   docker build -t ghcr.io/<owner>/image-processor:latest .
#   docker push ghcr.io/<owner>/image-processor:latest      # or: kind load docker-image ...
FROM python:3.12-slim

# Non-root runtime (matches k8s/deployment.yaml securityContext: runAsNonRoot + readOnlyRootFilesystem).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (better layer caching). requirements.txt lists `edgecommons`,
# resolved from the registry (PyPI or a pip git+https dep) — no monorepo checkout needed.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The entry point, the package and all its subpackages, and the schemas the component validates
# its own configuration, its result bodies, and every model manifest against at runtime.
COPY main.py /app/main.py
COPY image_processor /app/image_processor
COPY config.schema.json /app/config.schema.json
COPY schemas /app/schemas

# The durable state, the model cache, and the staging tree the component owns. A deployment
# mounts a volume over /var/lib/edgecommons so the ledger and the cache outlive the container,
# and mounts the camera spool read-write wherever the routes name it.
RUN mkdir -p /var/lib/edgecommons/image-processor \
    && chown -R 65532:65532 /var/lib/edgecommons

# Drop to a non-root UID; /tmp is provided writable by the Deployment (emptyDir).
USER 65532:65532

ENTRYPOINT ["python3", "/app/main.py"]
# No default args: with --platform auto the library detects KUBERNETES from the SA token
# (config source -> CONFIGMAP at /etc/edgecommons, transport -> MQTT, identity -> Downward API).
