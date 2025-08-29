
## Docker Hub for uploading images
- Create account on docker hub. 
- [Create a personal access token](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/unauthorized-docker-create-access-token-dockerhub-incorrect-username-password) 
- `docker login -u <USER_NAME>`. Copy the personal access token you created, not the password.
- `docker build . --tag <USER_NAME>/simple_docker_image`
- `docker push  <USER_NAME>/simple_docker_image`