FROM python:3.9-slim
COPY . /docker_app
WORKDIR /docker_app
RUN pip install -r requirement.txt
RUN git clone https://github.com/tanmoyie/Bayesian-Inference-Modeling/tree/clean-code
# RUN mkdir ~/.streamlit
# RUN cp config.toml ~/config.toml
# RUN cp credentials.toml ~/credentials.toml
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "BIMReTA_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD ["BIMReTA_app.py"]

# Reference https://docs.streamlit.io/deploy/tutorials/docker#check-network-port-accessibility