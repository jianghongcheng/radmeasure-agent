FROM python:3.12-slim
WORKDIR /app
COPY cases.json /app/cases.json
RUN useradd --create-home --uid 1000 agent && chown -R agent:agent /app
USER agent
