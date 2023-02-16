# Crop segmentation & classification

## Description

This repository is a backend for crop segmentation & classification project for Madagascar crops

## How to contribute

A service account is used to communicate with GEE (as described [here](https://developers.google.com/earth-engine/guides/service_account))

- Clone the repository
- Copy your service account's private key in root folder as ".private-key.json" (notice the "." prefix)
- Set your service account's mail to an environment variable called "SERVICE_ACCOUNT"
    PowerShell: $env:SERVICE_ACCOUNT="your-service-account@mail.com"
- Create a virtual environment and activate it
- Install requirements : `pip install -r requirements.txt`
- Run server : `python main.py`