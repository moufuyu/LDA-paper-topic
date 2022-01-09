# 概要  
科学系の論文が集約されているWebサイトであるarXivから、AIに関する論文の要約文を取得し、LDA（潜在ディクレ配分）でトピックを分類する。  

# 実行結果  
python get_topic.pyでプログラムを実行すると、5つのトピックとそのトピックを特徴づける上位5つの単語、その単語列に最も適合する論文の要約文(300字まで)を出力する。  
結果を見ると、graphやagentといった機械学習の各分野でよく使われる語が得られていることが確認できる。  

<img width="960" alt="image" src="https://user-images.githubusercontent.com/62968285/148703937-6da1d258-49ee-429c-9b9b-55f4f17b1cf1.png">
