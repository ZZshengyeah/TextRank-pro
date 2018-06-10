
echo "install nltk_data"
cd ../
chmod -R 777 textRank-en/
cd textRank-en
echo "copy file it may take a while"
cp -r nltk_data ~

cd src
echo "install dependent package"
pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/

cd word2vec

echo "Decompression data file"
unzip filter.zip

echo "train wor2vec model it may take one hour or more"
python train.py

echo "test improve textrank"
cd ../

python test_single.py
python test_csv_input.py

echo "finish"
