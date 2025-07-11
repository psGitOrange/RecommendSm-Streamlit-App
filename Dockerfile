# using python base image
FROM python:3.12-slim

# setup working directory in docker container
WORKDIR /app

# copy current directory contents to docker container at /app
COPY . .

# install requirements inside docker container
RUN pip install -r requirements.txt

# expose port of docker container
EXPOSE 8501

# command to run the app
CMD ["streamlit", "run", "main.py", "--server.port=8501"]