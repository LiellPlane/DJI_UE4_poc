# Define custom function directory
ARG FUNCTION_DIR="/function"

FROM python:3.9 as build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR


# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev

RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    ffmpeg \ 
    libsm6 \
    libxext6 \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN apt-get install libgl1
# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
COPY app/ ${FUNCTION_DIR}
#COPY --chown=user:user --chmod=600 app/ ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}

#RUN pip install -r requirements.txt --target ${FUNCTION_DIR}
RUN pip install -r requirements.txt --target ${FUNCTION_DIR}

#RUN export PYTHONPATH=$PYTHONPATH:/${FUNCTION_DIR}/scambilight/libs

# Install the function's dependencies
RUN pip install \
    --target ${FUNCTION_DIR} \
        awslambdaric


FROM python:3.9

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
#CMD [ "lambda_function.handler" ]
CMD ["DJI_UE4_poc/Source/scambilight/scambiloop.handler"]