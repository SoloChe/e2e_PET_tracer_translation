#!/bin/sh

image=$1

docker run -v $(pwd)/test:/opt/ml -p 8080:8080 --rm ${image} serve