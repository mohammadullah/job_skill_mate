#!/usr/bin/env bash

RESUME_FOLDER=${1}
ONET_JOB_TITLE=${2}

CALLING_FOLDER=$(pwd)

spark-submit --archives $CALLING_FOLDER/corpora.zip#corpora,$CALLING_FOLDER/tokenizers.zip#tokenizers --py-files $CALLING_FOLDER/lib.zip --num-executors 6 $CALLING_FOLDER/src/job_title_normalization_er.py file://$CALLING_FOLDER/$RESUME_FOLDER file://$CALLING_FOLDER/$ONET_JOB_TITLE