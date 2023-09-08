FROM ubuntu:22.04

WORKDIR /app

COPY ./requirements.txt ./
RUN apt-get update && apt-get install -y python3 python3-pip
#this line only needed for specific containers that may not work with opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
#COPY ./src ./src
#EXPOSE 8080

#CMD ["flask","--app", "./src/hello" ,"--debug","run","--host=0.0.0.0", "--port=8080" ]
