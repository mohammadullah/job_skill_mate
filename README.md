# Skills Job Advisor

This is a Project work of Big Data Programming II (SFU, CMPT 733) by Btara Truhandarien, Mohammad W. Ullah, Bhuvan Chopra and Grace Kingsly. 


# Datasets, raw findings and additional data
You can find them in this [Google Drive](https://drive.google.com/drive/u/0/folders/1-glxM4PAj-fo3U4Dt2O3ro3RwTeVj8_V) folder (request permission if needed)

These datasets come from Indeed resume repository which was scraped using this [scraping tool](https://github.com/btruhand/indeed-resume-scraper) that we also built ourselves.

Data should not contain any private details (but they may accidentally do, considering the unstructured nature of resumes). We ask of you not to share or make the data public, thank you.

# Cleaning data

## Resume data

In order to clean resume data, make sure that you have them all in one directory. Then you can invoke the following
```bash
sh scripts/clean_dataset.sh <path to directory holding resume data>
```

The script file will automatically make a `cleaned` directory under the given directory and the cleaned resumes will be put there as JSON files

## Cleaning O-NET Job Titles
To clean O-NET job title (used for job title ER/normalization) then just run the following

```bash
python -m src.clean_alternate_titles onet/onet_alternate_titles.csv onet/cleaned_onet_alternate_titles.json
gzip onet/cleaned_onet_alternate_titles.json
```

The alternate titles willn lower case and removed of punctuations and stopwords.

# Running Job Title Normalization (ER)
This has only been tested on the SFU cluster but please perform the following steps (assuming you've uploaded the project to the gateway)

First go to the SFU cluster and download the punkt and stopwords nltk data
```bash
ssh gateway.sfucloud.ca
module load spark
module load pypy3

# just download all
python -m nltk.downloader all
```

Now zip the `corpora` and `tokenizers` folders
```bash
cd ~/nltk_data/corpora
zip -r corpora.zip .
mv corpora.zip <path to project directory>

cd ~/nltk_data/tokenizers
zip -r tokenizers.zip .
mv tokenizers.zip <path to project directory>
```

Now in the project directory
```bash
# zip lib directory
zip -r lib.zip lib
```
You should do the above everytime you change the relevant parts in lib.zip

Now you can just run (make sure you've done `module load spark` and `module load pypy3`)
```bash
sh scripts/run_job_title_er.sh <relative path to resume dataset> onet/cleaned_onet_alternate_titles.json.gz
```

This will take quite a while
