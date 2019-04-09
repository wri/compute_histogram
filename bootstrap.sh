add-apt-repository ppa:ubuntugis/ppa && apt-get update && apt-get -y upgrade
apt-get --yes install python3.6-dev gdal-bin libgdal-dev python3-pip ca-certificates git
mkdir -p /etc/pki/tls/certs
cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt
cd ~
git clone https://github.com/wri/compute_histogram.git
cd compute_histogram
pip3 install -e .