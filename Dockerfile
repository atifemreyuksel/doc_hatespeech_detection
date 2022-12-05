FROM continuumio/miniconda3:4.12.0 as base
WORKDIR /app

COPY hs_environment.yml .
RUN conda env update --file hs_environment.yml

EXPOSE 3442
COPY . .

RUN mkdir -p checkpoints/rulemodel
RUN wget -q https://tulap.cmpe.boun.edu.tr/staticFiles/hatespeech_detection/config_20220817220406312.json -O checkpoints/rulemodel/config_20220817220406312.json
RUN wget -q https://tulap.cmpe.boun.edu.tr/staticFiles/hatespeech_detection/best_model.pth -O checkpoints/rulemodel/best_model.pth

SHELL ["conda", "run", "-n", "nlp_env", "/bin/bash", "-c"]
RUN pip install uvicorn fastapi

ENTRYPOINT [ "conda", "run", "-n", "nlp_env", "uvicorn", "app:app", "--port", "3442", "--host", "0.0.0.0"  ]
