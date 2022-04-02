FROM python:3.10-slim-buster
WORKDIR /tcplaygroung
ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
COPY tokenizer.pickle tokenizer.pickle
COPY nonverbal nonverbal
COPY vectors.txt vectors.txt
COPY W_w2v W_w2v
COPY W_glove W_glove
COPY word2vec.kv word2vec.kv
COPY w2v_dict.p w2v_dict.p
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run"]
