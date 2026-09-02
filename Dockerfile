FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLAUSEFORGE_HOST=0.0.0.0 \
    CLAUSEFORGE_PORT=8000 \
    CLAUSEFORGE_MODEL_BACKEND=mock
COPY --from=builder /opt/venv /opt/venv
RUN addgroup --system clauseforge && adduser --system --ingroup clauseforge clauseforge
USER clauseforge
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
ENTRYPOINT ["uvicorn", "clauseforge.serving.app:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
