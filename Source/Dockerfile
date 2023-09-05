FROM ubuntu:22.04

WORKDIR /app

COPY ./requirements_scambi.txt ./
RUN apt-get update && apt-get install -y python3 python3-pip
#this line only needed for specific containers that may not work with opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements_scambi.txt
#can't copy anyting out of docker scope
COPY ./scambilight ./scambilight
COPY ./lumotag ./lumotag
EXPOSE 8080

CMD ["sudo", "python3", "./src/scambilight/scambiloop.py"]
