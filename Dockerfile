FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY prompts/ prompts/
COPY glossaries/ glossaries/

RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["python3", "-m", "idraak.cli", "demo"]
