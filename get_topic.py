### 論文が多数公開されているarXivから各論文の要約文を取得し、LDAを用いてトピックを生成する
import feedparser
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# arXivから論文情報を取得
def get_paper():
    d = feedparser.parse('http://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=500&sortBy=lastUpdatedDate')
    entries = d['entries']
    abst_list = []
    for num in range(500):
        abst = entries[num].summary
        # 改行文字の削除
        abst = abst.replace('\n' , ' ')
        abst_list.append(abst)

    return abst_list

# 潜在ディクレ配分（LDA）を行う。入力はBoW行列。分析結果を出力として返す。
def LDA(text_list, topic_num = 5):
    # BoW行列の作成

    # max_dfは最大文書頻度で、さまざまな文書に出現しすぎている語を除外する。
    # max_featuresは出現頻度が最も高いとみなされる単語の語数を指定する。
    count = CountVectorizer(stop_words='english', max_df = 0.175, max_features = 5000)
    # BoWモデルの語彙を生成
    BoW = count.fit_transform(text_list)

    #LDAの実行
    # 10トピックを生成
    lda = LatentDirichletAllocation(n_components=topic_num, random_state=0, learning_method='batch')
    topics = lda.fit_transform(BoW)

    #各トピックを特徴付ける単語の出力
    n_top_words = 5
    feature_names = count.get_feature_names()

    for topic_idx, topic in enumerate(lda.components_):
        print('トピック %d:' % (topic_idx + 1))
        # [:,:,-1]で降順にソート。
        print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1: -1]])) 
        print('このトピックに最も適合しているsummary:')
        print(text_list[np.argmax(topics[:, topic_idx])][:300])
        print()


text_list = get_paper()
LDA(text_list)

        






