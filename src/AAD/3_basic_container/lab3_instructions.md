## Lab 3: Build & run docker
- Step 1: `cd docker/lab\ 1\ -\ simple\ container` or `cd lab<TAB>1<TAB> to use AUTOCOMPLETE IN BASH`
- Step 2: Building docker image - `docker build . --tag first-docker-image`
- Step 3: Run docker container - `docker run -d -p 5000:5000 first-docker-image` or `docker run -d -p 127.0.0.1:5000:5000 first-docker-image`. Port numbers must match at all 3 places. Dockerfile, Python Source Code & Docker Run command