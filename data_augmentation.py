import random
import nltk
from nltk.corpus import wordnet, stopwords
from transformers import pipeline

def augment_text(text: str, prob: float = 0.7) -> str:
    if random.random() > prob:
        return text

    # words = nltk.word_tokenize(text)
    new_words = []
    # stop_words = set(stopwords.words('english'))

    # 同义词替换增强
    # for i, word in enumerate(words):
    #     if word.lower() in stop_words or len(word) < 3:
    #         new_words.append(word)
    #         continue

    #     synonyms = []
    #     for syn in wordnet.synsets(word):
    #         for lemma in syn.lemmas():
    #             if lemma.name() != word and "_" not in lemma.name():
    #                 synonyms.append(lemma.name())

    #     # 提升替换概率到70%并增加多位置换
    #     if synonyms and random.random() < 0.7:
    #         new_words.append(random.choice(synonyms))
    #         if random.random() < 0.3:  # 30%概率添加第二个同义词
    #             new_words.append(random.choice(synonyms))
    #     else:
    #         new_words.append(word)

    # # 扩展并平衡情感词库
    # emotion_words = {
    #     'positive': ['joy', 'happy', 'wonderful', 'excellent', 'fantastic', 
    #                 'brilliant', 'peace', 'love', 'amazing', 'victory',
    #                 'delight', 'gratitude', 'harmony', 'success'],
    #     'negative': ['anger', 'hate', 'fear', 'sadness', 'disaster',
    #                 'tragedy', 'war', 'fury', 'destruction', 'nuke',
    #                 'threatening', 'hostile', 'violence', 'conflict']
    # }

    # # 情感词插入并平衡正负面
    # if new_words and random.random() < 0.8:
    #     emotion_type = random.choice(['positive', 'negative'])
    #     for _ in range(random.randint(1, 2)):  # 插入1-2个情感词
    #         new_words.insert(random.randint(0, len(new_words)), 
    #                        random.choice(emotion_words[emotion_type]))

    # 添加随机词语删除增强
    if len(new_words) > 4:
        del_count = random.randint(1, 20)  # 每次删除1-20个词
        for _ in range(del_count):
            if len(new_words) > 3:
                del_idx = random.randint(0, len(new_words)-1)
                del new_words[del_idx]

    # 添加对抗性扰动
    adversarial_words = ['however', 'but', 'although', 'yet', 'except']
    if random.random() < 0.4:
        new_words.insert(random.randint(0, len(new_words)), random.choice(adversarial_words))

    # 局部语义替换
    if random.random() < 0.3:
        try:
            generator = pipeline('text-generation', model='gpt2',
                               max_new_tokens=50,  # 改为生成新token数
                               truncation=True,  # 添加截断参数
                               device=-1)  # 强制使用CPU时不输出设备信息
            # 限制输入长度（保留生成空间）
            input_text = ' '.join(new_words[:1500])  # 取前1500个词作为输入
            rewritten = generator(input_text, num_return_sequences=1)[0]['generated_text']
            new_words = nltk.word_tokenize(rewritten)
        except Exception as e:  # 捕获更广泛的异常
            # print(f"GPT-2生成失败: {str(e)}")
            pass


    return ' '.join(new_words)