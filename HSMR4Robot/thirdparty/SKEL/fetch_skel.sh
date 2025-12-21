#!/bin/bash
# Installation script inspired by the excellent DECA code repository : https://github.com/yfeng95/DECA

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


#  Fetch SKEL data
echo -e "\nBefore you continue, you must register at https://skel.is.tue.mpg.de/ and agree to the SKEL license terms."
read -p "Username (SKEL):" username
read -p "Password (SKEL):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p ./data
echo -e "\nDownloading SKEL..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=skel&resume=1&sfile=skel_models_v1.1.zip&resume=1' -O './data/SKEL.zip' --no-check-certificate --continue

#check that the file is more than 100MB
if [ ! -f ./data/SMPLH.tar.xz ] || [ $(stat -c%s ./data/SMPLH.tar.xz) -lt 100000000 ]; then
    echo "Error: SMPLH download failed or file is too small. Please check your credentials and try again."
    exit 1
fi

# Unzip and place in the right directory
unzip ./data/SKEL.zip -d ./data/SKEL
mv ./data/SKEL/skel_models_v1.1/ ./data/skel/ # move the model's file to be under ./data/skel/
rm -rf ./data/SKEL
rm ./data/SKEL.zip
