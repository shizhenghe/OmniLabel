# Upstream supplies ROS2, Ceres, GTSAM, Iridescence and the real calibrate binary.
# Pin BASE_IMAGE to an audited sha256 digest for a reproducible deployment.
ARG BASE_IMAGE=koide3/direct_visual_lidar_calibration:humble
FROM ${BASE_IMAGE}
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends xvfb xauth libgl1-mesa-dri libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*
# --background hides the viewer but upstream still creates an OpenGL context.
ENV LIBGL_ALWAYS_SOFTWARE=1
# Inherit upstream /ros_entrypoint.sh, which sources both ROS and Koide's workspace.
CMD ["xvfb-run", "-a", "ros2", "run", "direct_visual_lidar_calibration", "calibrate", "/calibration", "--background", "--auto_quit"]
