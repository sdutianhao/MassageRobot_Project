#!/bin/bash
# Installation script inspired by the excellent DECA code repository : https://github.com/yfeng95/DECA

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

#Fetch SMPL data
echo -e "\nBefore you continue, you must register at https://mano.is.tue.mpg.de and agree to the MANO license terms."
read -p "Username (MANO):" username
read -p "Password (MANO):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p ./data
# Download SMPLH
echo -e "\nDownloading SMPLH..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz' -O './data/SMPLH.tar.xz' --no-check-certificate --continue
mkdir -p ./data/SMPLH

#check that the file is more than 100MB
if [ ! -f ./data/SMPLH.tar.xz ] || [ $(stat -c%s ./data/SMPLH.tar.xz) -lt 100000000 ]; then
    echo "Error: SMPLH download failed or file is too small. Please check your credentials and try again."
    exit 1
fi

tar -xf ./data/SMPLH.tar.xz -C ./data/SMPLH 

# Download MANO
echo -e "\nDownloading MANO..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip' -O './data/MANO.zip' --no-check-certificate --continue
mkdir -p ./data/MANO

#check that the file is more than 100MB
if [ ! -f ./data/SMPLH.tar.xz ] || [ $(stat -c%s ./data/SMPLH.tar.xz) -lt 100000000 ]; then
    echo "Error: SMPLH download failed or file is too small. Please check your credentials and try again."
    exit 1
fi

unzip -q ./data/MANO.zip -d ./data/MANO

# Merge MANO to smplh as instructed in https://github.com/vchoutas/smplx/blob/main/tools/README.md

mkdir -p ./data/smplx/smplh

python dependancies/merge_smplh_mano.py \
--smplh-fn data/SMPLH/female/model.npz  \
--mano-left-fn data/MANO/mano_v1_2/models/MANO_LEFT.pkl \
--mano-right-fn data/MANO/mano_v1_2/models/MANO_RIGHT.pkl \
--output-folder data/smplx/smplh

mv data/smplx/smplh/model.pkl data/smplx/smplh/SMPLH_FEMALE.pkl

python dependancies/merge_smplh_mano.py \
--smplh-fn data/SMPLH/male/model.npz  \
--mano-left-fn data/MANO/mano_v1_2/models/MANO_LEFT.pkl \
--mano-right-fn data/MANO/mano_v1_2/models/MANO_RIGHT.pkl \
--output-folder data/smplx/smplh

mv data/smplx/smplh/model.pkl data/smplx/smplh/SMPLH_MALE.pkl

