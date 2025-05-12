import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset_loading import load_dataset
from data_augmentation import augment_text

# 初始化深度学习模型（新增部分）
def init_bert_model():
    model_path = "E:/MyOwn/ProgramStudy/NLP/distilbert-base-uncased-finetuned-sst-2-english"
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model

def analyze_sentiments_bert(data: dict, ifAugment: bool) -> dict:
    """基于DistilBERT的情感分析"""
    tokenizer, model = init_bert_model()

    for post_data in data.values():
        if ifAugment:
            original_text = post_data['Comments']
            augmented_text = augment_text(original_text)
            # 改为概率混合模式（0.3概率使用纯增强文本）
            text = augmented_text if random.random() < 0.3 else f"{original_text} [AUG] {augmented_text}"
        else:
            text = post_data['Comments']

        # BERT预处理（新增部分）
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)

        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)

        # 解析结果（新增部分）
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        post_data['sentiment'] = {
            'NEGATIVE': probabilities[0][0].item(),
            'POSITIVE': probabilities[0][1].item()
        }

    return data

# 保持输出格式一致的导出函数
def export_analysis_results(data: dict, ifAugment: bool):
    if not data:
        print("没有可用的数据")
        return

    output_dir = os.path.join('outPut')
    os.makedirs(output_dir, exist_ok=True)

    output_fileName = 'distilBertAnalysisResult.txt' if not ifAugment else 'distilBertAnalysisResult_augment.txt'
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
            if sentiment['POSITIVE'] > 0.7:
                positive_count += 1
            elif sentiment['NEGATIVE'] > 0.7:
                negative_count += 1
            else:
                neutral_count += 1

            f.write(f"\n记录 {key}:")
            f.write(f"\n标题: {value['title']}")
            f.write("\n情感判断: " + 
                  ("积极" if value['sentiment']['POSITIVE'] > 0.7 else 
                   "消极" if value['sentiment']['NEGATIVE'] > 0.7 else "中性"))
            f.write("\n" + "-"*50)

        f.write("\n情感统计结果:")
        f.write(f"积极: {positive_count} 条 ({positive_count/total:.1%})")
        f.write(f"中性: {neutral_count} 条 ({neutral_count/total:.1%})")
        f.write(f"消极: {negative_count} 条 ({negative_count/total:.1%})")
        print(f"成功处理全部{total}条数据,结果已保存至./outPut/{output_fileName}")

if __name__ == "__main__":
    # 原始文本 → Tokenizer向量化 → 模型推理 → Logits输出 → Softmax量化 → 概率结果
    posts_data = load_dataset()

    processed_data = analyze_sentiments_bert(posts_data, ifAugment=False)
    export_analysis_results(processed_data, ifAugment=False)

    processed_Augmented_data = analyze_sentiments_bert(posts_data, ifAugment=True)
    export_analysis_results(processed_Augmented_data, ifAugment=True)
