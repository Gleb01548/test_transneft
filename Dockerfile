ARG VERSION=3.11-slim

FROM python:${VERSION}

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false
COPY pyproject.toml pdm.lock README.md /project/

WORKDIR /project
RUN pdm install --check --prod --no-editable

COPY src/ /project/src

ENV PATH="/project/.venv/bin:$PATH"
COPY src /project/src
COPY data /project/data
COPY conf.yaml /project/conf.yaml

CMD ["python", "-m", "src.app"]