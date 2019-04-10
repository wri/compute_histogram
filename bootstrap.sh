#! /bin/sh

add-apt-repository -y ppa:ubuntugis/ppa && apt-get -y update && apt-get -y upgrade
apt-get -y install python3.6-dev gdal-bin libgdal-dev python3-pip ca-certificates git awscli
mkdir -p /etc/pki/tls/certs
cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt
cd /home/ubuntu
git clone https://github.com/wri/compute_histogram.git
cd compute_histogram
pip3 install -e .
chmod -R 777 compute_histogram