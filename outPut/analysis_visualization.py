import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

def parse_result_file(file_path):
    """解析结果文件获取百分比数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        last_line = f.readlines()[-1].strip()
        
    # 使用正则表达式提取百分比
    pattern = r'积极: (\d+) 条 \(([\d.]+)%\).*?中性: (\d+) 条 \(([\d.]+)%\).*?消极: (\d+) 条 \(([\d.]+)%\)'
    match = re.search(pattern, last_line)
    
    return {
        '积极': float(match.group(2)),
        '中性': float(match.group(4)),
        '消极': float(match.group(6))
    }

def plot_sentiment_pie(results, titles):
    """绘制饼状图"""
    # 添加中文字体配置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 14  # 新增全局字体大小设置
    plt.rcParams['axes.titlesize'] = 16  # 标题字体
    
    plt.figure(figsize=(15, 10))
    
    colors = ['#66b3ff', '#99ff99', '#ff9999']
    labels = ['积极', '中性', '消极']
    
    for i, (title, data) in enumerate(zip(titles, results)):
        plt.subplot(2, 2, i+1)
        plt.pie(data.values(), 
               labels=labels, 
               colors=colors,
               autopct='%1.1f%%', 
               startangle=90,
               textprops={'fontsize': 14})  # 添加文本大小参数
        plt.title(title, fontsize=16)  # 调整标题字体大小
    
    plt.tight_layout()
    plt.savefig('./outPut/sentiment_distribution.png')
    plt.close()
    print("可视化结果已保存至 ./outPut/sentiment_distribution.png")

if __name__ == "__main__":
    # 需要分析的文件列表
    files = [
        'tfIdfSentimentAnalysisResult.txt',
        'tfIdfSentimentAnalysisResult_augment.txt',
        'distilBertAnalysisResult.txt', 
        'distilBertAnalysisResult_augment.txt'
    ]
    
    # 解析所有文件
    results = []
    for file in files:
        file_path = os.path.join('outPut', file)
        results.append(parse_result_file(file_path))
    
    # 生成可视化图表
    plot_sentiment_pie(results, [
        'TF-IDF 基础版', 
        'TF-IDF 增强版',
        'DistilBERT 基础版',
        'DistilBERT 增强版'
    ])