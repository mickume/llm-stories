#!/bin/bash

# Usage: ./bin/train.sh 'bigscience/bloom-3b' 'mickume/harry_potter_small' 'token' config

MODEL=$1
REPO=$2
TOKEN=$3
CONFIG_PATH=$4

BUCKET=llm-stories
PACKAGE_NAME=trainer-1
REGION=europe-west4

DATE=`date '+%Y%m%d_%H%M%S'`
JOB_NAME=train_
JOB_ID=$JOB_NAME$DATE
PACKAGE_PATH=gs://$BUCKET/packages/$PACKAGE_NAME.tar.gz
JOB_DIR=gs://$BUCKET/jobs/$JOB_ID

# launch the job
gcloud ai-platform jobs submit training $JOB_ID \
    --job-dir $JOB_DIR \
    --region $REGION \
    --python-version '3.7' \
    --runtime-version '2.11' \
    --packages $PACKAGE_PATH \
    --module-name 'trainer.train' \
    --config $CONFIG_PATH \
    -- \
    --model-name $MODEL \
    --repo $REPO \
    --token $TOKEN