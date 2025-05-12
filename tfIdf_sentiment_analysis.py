import os
from dataset_loading import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from data_augmentation import augment_text

def analyze_sentiments(data: dict, ifAugment: bool) -> dict:
    # 提取所有Comments文本内容
    corpus = []
    for item in data.values():
        text = item['Comments']
        if ifAugment:
            text = augment_text(text)  # 应用数据增强
        corpus.append(text)

    # 初始化TF-IDF转换器
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000  # 限制特征数量
    )

    # 训练模型并转换数据
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 将结果添加回原始数据结构
    for idx, comments_data in data.items():
        comments_data['tfidf'] = tfidf_matrix[idx]

    # 情感分析
    sid = SentimentIntensityAnalyzer()
    for comments in data.values():
        text = comments['Comments']
        if ifAugment:
            text = augment_text(text)  # 应用数据增强
        sentiment = sid.polarity_scores(text)
        comments['sentiment'] = sentiment['compound']

    return data

def export_analysis_results(data: dict, ifAugment: bool):
    if not data:
        print("没有可用的数据")
        return

    output_dir = os.path.join('outPut')
    os.makedirs(output_dir, exist_ok=True)

    output_fileName = 'tfIdfSentimentAnalysisResult.txt' if not ifAugment else 'tfIdfSentimentAnalysisResult_augment.txt'
    output_path = os.path.join(output_dir, output_fileName)
    with open(output_path, 'w', encoding='utf-8') as f:
        total = len(data)
        positive_count = 0
        neutral_count = 0
        negative_count = 0

        f.write(f"数据集包含 {total} 条记录\n")

        # 使用迭代器遍历所有数据
        for idx, (key, value) in enumerate(data.items()):
            if idx >= total:
                break

            sentiment = value['sentiment']
            if sentiment > 0.05:
                positive_count += 1
            elif sentiment < -0.05:
                negative_count += 1
            else:
                neutral_count += 1

            f.write(f"\n记录 {key}:")
            f.write(f"\n标题: {value['title']}")
            f.write(f"\nTF-IDF特征数量: {len(value['tfidf'].data)}")
            f.write(f"\n情感得分: {value['sentiment']:.2f}")
            f.write("\n情感判断: " + 
                  ("积极" if value['sentiment'] > 0.1 else 
                   "消极" if value['sentiment'] < -0.1 else "中性"))
            f.write("\n" + "-"*50)

        f.write("\n情感统计结果:")
        f.write(f"积极: {positive_count} 条 ({positive_count/total:.1%})")
        f.write(f"中性: {neutral_count} 条 ({neutral_count/total:.1%})")
        f.write(f"消极: {negative_count} 条 ({negative_count/total:.1%})")
        print(f"成功处理全部{total}条数据,结果已保存至./outPut/{output_fileName}")

if __name__ == "__main__":
    posts_data = load_dataset()

    processed_data = analyze_sentiments(posts_data, ifAugment=False)
    export_analysis_results(processed_data, ifAugment=False)

    processed_Augmented_data = analyze_sentiments(posts_data, ifAugment=True)
    export_analysis_results(processed_Augmented_data, ifAugment=True)