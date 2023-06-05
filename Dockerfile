FROM continuumio/anaconda3
COPY . /usr/app/
EXPOSE 8501
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD streamlit run FLaskApp_frontend_Streamlit.py