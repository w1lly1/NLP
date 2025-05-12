import csv
from pathlib import Path
import os

def load_dataset() -> dict:
    """加载并结构化数据集"""
    # 增加CSV字段大小限制
    max_int = 2147483647  # Windows平台C long的最大值
    csv.field_size_limit(max_int)
    
    dataset_path = Path(__file__).parent / 'dataSet' / 'reddit_trump.csv'
    posts = {}

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            i=0
            for row in reader:
                if not row.get('Key'):
                    continue
                
                # 简化元数据解析
                posts[i] = {
                    "key": row['Key'],
                    "Id": row['Id'],
                    "title": row['Title'].strip(),
                    "post": row['Post'].strip(),
                    "post_date": row['Post Date'],
                    "meta": row['Meta'],
                    "Comments": row['Comments']
                }
                i+=1
    
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return {}
    
    return posts


def print_data(data: dict, limit: int = 10) -> None:
    """打印数据预览"""
    if not data:
        print("没有可用的数据")
        return

    print(f"数据集包含 {len(data)} 条记录")


    output_dir = os.path.join('outPut')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'dataSample.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"数据集包含 {len(data)} 条记录\n")
        
        f.write("\n第一组数据:\n")
        # 直接获取第一条数据
        first_key = next(iter(data))
        value = data[first_key]
        
        f.write(f"\nRecord {first_key}:\n")
        f.write(f"Key: {value['key']}\n")
        f.write(f"Id: {value['Id']}\n")
        f.write(f"Title: {value['title']}\n")
        f.write(f"Post: {value['post']}\n")
        f.write(f"Post Date: {value['post_date']}\n")
        f.write(f"Meta: {value['meta']}\n")
        f.write(f"Comments: {value['Comments']}\n")
        f.write("-" * 50 + "\n")

    # print("\n数据预览:")
    # for key, value in list(data.items())[:limit]:
    #     print(f"\nRecord {key}:")
    #     print(f"Key: {value['key']}")
    #     print(f"Id: {value['Id']}")
    #     print(f"Title: {value['title']}")
    #     print(f"Post: {value['post']}")
    #     print(f"Post Date: {value['post_date']}")
    #     print(f"Meta: {value['meta']}")
    #     print(f"Comments: {value['Comments']}")
    #     print("-" * 50)


if __name__ == "__main__":
    posts_data = load_dataset()
    print_data(posts_data)