# syntax=docker/dockerfile:1
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=secret,id=proxy_ca,required=true \
    apt-get -o Acquire::https::CaInfo=/run/secrets/proxy_ca update && \
    apt-get -o Acquire::https::CaInfo=/run/secrets/proxy_ca install -y --no-install-recommends \
      ca-certificates cmake g++ libopencv-dev libpcl-dev libboost-filesystem-dev libjsoncpp-dev && \
    rm -rf /var/lib/apt/lists/*
COPY auto_calib /opt/pjlab/auto_calib
# Ubuntu 22.04 ships shared JsonCpp; upstream hardcodes libjsoncpp.a.
# Change only the link target, keeping the calibration algorithm unchanged.
RUN sed -i 's/libjsoncpp.a/jsoncpp/g' /opt/pjlab/auto_calib/CMakeLists.txt && \
    cmake -S /opt/pjlab/auto_calib -B /opt/pjlab/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/pjlab/build --parallel 2
LABEL org.opencontainers.image.source="https://github.com/PJLab-ADG/SensorsCalibration" \
      org.opencontainers.image.revision="7fe8d46caac040cbcd89cc65e79e5519df01bcd4"
ENTRYPOINT ["/opt/pjlab/auto_calib/bin/run_lidar2camera"]
