FROM python:3

WORKDIR /answer_question_model

COPY . .

# RUN pip install --no-cache-dir -r req.txt
RUN pip install torch
RUN pip install torchtext
RUN pip install pandas
RUN pip install transformers
RUN pip install sentencepiece
RUN pip install numpy
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install streamlit
RUN pip install streamlit-option-menu