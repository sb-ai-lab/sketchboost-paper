source sb_env/bin/activate

pip install -r requirements.txt
ipython kernel install --user --name=sb_env

# install GBDTMO

git clone https://github.com/zzd1992/GBDTMO.git

cd GBDTMO
bash make.sh
pip install .
cd ..