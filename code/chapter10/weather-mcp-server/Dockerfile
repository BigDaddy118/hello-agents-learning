# Weather MCP Server — Streamable HTTP for Smithery
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY server.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PORT=8081
ENV MCP_TRANSPORT=streamable-http
ENV HOST=0.0.0.0

EXPOSE 8081

CMD ["python", "server.py"]
