# This is the [Bolshoi Plank](https://hipacc.ucsc.edu/Bolshoi/index.html) Simulation

**TABLE OF CONTENTS:**

- [This is the Bolshoi Plank Simulation](#this-is-the-bolshoi-plank-simulation)
  - [Download the datasets](#download-the-datasets)
  - [Setting Up](#setting-up)

## Download the datasets

(You will need an account in CosmoSim for this, register
[here](https://www.cosmosim.org/auth/registration/register?redirect=/auth/login))
The data for the simulation can be downloaded
[here](https://www.cosmosim.org/cms/files/rockstar-data/)

You will need to choose BolshoiP -> Catalogs -> `hlist_1.00231.list.gz`
(for z = 1 data)

We can download the data using the terminal like this:

```bash
wget --user=USER --ask-password --auth-no-challenge $url
```

`$url` is the url you get with `hlist_1.00231.list.gz`

## Setting Up

First, create a conda env with `requirements.txt` file provided to get all the
required libraries using this command:

```bash
conda create --name <env> --file requirements.txt
conda activate <env>
```

Make sure to download the dataset and change your `ABS_PATH` in `constants.py`
to the folder where you downloaded the halos file.

Then run the preprocessing script using this command:

```bash
python preprocess.py
```
