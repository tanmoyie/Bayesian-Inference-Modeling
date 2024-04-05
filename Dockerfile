FROM python:3.11
COPY . /docker_app
WORKDIR /docker_app
RUN pip install -r requirement.txt
# RUN mkdir ~/.streamlit
# RUN cp config.toml ~/config.toml
# RUN cp credentials.toml ~/credentials.toml
EXPOSE $PORT
ENTRYPOINT ["streamlit", "run"]
CMD ["BIMReTA_app.py"]
