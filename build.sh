#!bin/bash

# install docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh
rm get-docker.sh

# start up docker
sudo pkill -9 dockerd
sudo rm -f /var/run/docker.pid
sudo dockerd & > /dev/null
sleep 5
echo "====== docker server started ======"
sudo chmod 666 /var/run/docker.sock
docker images

# install deps for front-end app
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash

# build and run image
git pull
sudo docker rm -f mycontainer
sudo docker image rm -f myimage
sudo docker build --no-cache -t myimage .
sudo docker run -d --name mycontainer -p 443:443 myimage
# sudo uvicorn app.main:app --host 0.0.0.0 --port 443 --ssl-keyfile privkey.pem --ssl-certfile fullchain.pem
