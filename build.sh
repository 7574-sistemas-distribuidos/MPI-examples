#!/bin/bash
cd base_image && \
docker build -f Dockerfile -t openmpi:v0.0.1 .
