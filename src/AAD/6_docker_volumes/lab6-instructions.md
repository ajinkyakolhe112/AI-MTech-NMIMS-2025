#### Volume Mount Lab
- `docker build . -t volumes_container`
- `docker run -v local_directory:/mounted_directory_in_container -p5000:5000 volumes_container`
- check if folder's content is read by flask api
- change file content and see if flask api reads updated file contet.
- alternative `docker-compose up --build`

- create docker volume by `docker volume create mounted_volume`
- run docker container with mountaed volume `docker run -v mounted_volume:/data -it ubuntu`
