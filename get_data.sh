# get experimental code
source sb_env/bin/activate

# get GBDT-MO experimental code
git clone https://github.com/zzd1992/GBDTMO-EX.git

# download datasets
gdown 1JBoyc5neSZExcs5qcpk73We9wOvPbXs4
unzip -o datasets.zip

gdown 0B3lPMIHmG6vGdG1jZ19VS2NWRVU
unzip -o Delicious.zip -d data
gdown 0B3lPMIHmG6vGY3B4TXRmZnZBTkk
unzip -o Mediamill.zip -d data

rm datasets.zip Delicious.zip Mediamill.zip

wget https://www.openml.org/data/get_csv/19335692 -O data/helena.csv
wget https://www.openml.org/data/get_csv/19335690 -O data/dionis.csv

wget https://www.openml.org/data/get_csv/21230443 -O data/scm20d.csv
wget https://www.openml.org/data/get_csv/21230440 -O data/rf1.csv

# process data
cp data/nus-wide/nus-wide-full-cVLADplus-test.arff GBDTMO-EX/dataset/
cp data/nus-wide/nus-wide-full-cVLADplus-train.arff GBDTMO-EX/dataset/
cd GBDTMO-EX
python loader.py
cd ..
python preprocess.py
