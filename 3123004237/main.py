import sys
import jieba
import math
from collections import Counter
import argparse
import numpy as np
from typing import List


def read_file(file_path):
    """
    读取文件内容，尝试多种编码格式
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'big5']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise Exception(f"文件不存在: {file_path}")
        except IOError:
            raise Exception(f"无法读取文件: {file_path}")

    raise Exception(f"无法解码文件: {file_path}")


def write_file(file_path, content):
    """
    写入文件内容
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except IOError:
        raise Exception(f"无法写入文件: {file_path}")


def preprocess_text(text: str) -> List[str]:
    """
    文本预处理：分词并过滤停用词
    """
    # 这里可以添加更多停用词
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
                 '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    words = jieba.cut(text)
    return [word for word in words if word.strip() and word not in stopwords]


def cosine_similarity_numpy(text1: str, text2: str) -> float:
    """
    使用NumPy优化计算两段文本的余弦相似度
    """
    # 延迟导入NumPy
    try:
        import numpy as np
    except ImportError:
        print("警告: NumPy未安装，使用原生Python计算余弦相似度")
        return cosine_similarity_fallback(text1, text2)

    # 预处理文本
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    if not words1 or not words2:
        return 0.0

    # 创建词频计数器
    counter1 = Counter(words1)
    counter2 = Counter(words2)

    # 获取所有词汇的并集
    all_words = list(set(counter1.keys()).union(set(counter2.keys())))

    # 创建词频向量
    vector1 = np.array([counter1.get(word, 0) for word in all_words], dtype=np.float32)
    vector2 = np.array([counter2.get(word, 0) for word in all_words], dtype=np.float32)

    # 计算点积
    dot_product = np.dot(vector1, vector2)

    # 计算模长
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # 避免除以零
    if magnitude1 * magnitude2 == 0:
        return 0.0

    # 计算余弦相似度
    return dot_product / (magnitude1 * magnitude2)


def cosine_similarity_fallback(text1: str, text2: str) -> float:
    """
    回退方案：使用原生Python计算余弦相似度
    """
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    if not words1 or not words2:
        return 0.0

    # 创建词频计数器
    counter1 = Counter(words1)
    counter2 = Counter(words2)

    # 获取所有词汇的并集
    all_words = set(counter1.keys()).union(set(counter2.keys()))

    # 创建词频向量
    vector1 = [counter1.get(word, 0) for word in all_words]
    vector2 = [counter2.get(word, 0) for word in all_words]

    # 计算点积
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

    # 计算模长
    magnitude1 = math.sqrt(sum(v * v for v in vector1))
    magnitude2 = math.sqrt(sum(v * v for v in vector2))

    # 避免除以零
    if magnitude1 * magnitude2 == 0:
        return 0.0

    # 计算余弦相似度
    return dot_product / (magnitude1 * magnitude2)


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    计算两段文本的Jaccard相似度
    """
    words1 = set(preprocess_text(text1))
    words2 = set(preprocess_text(text2))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union != 0 else 0.0


def weighted_jaccard_similarity(text1: str, text2: str) -> float:
    """
    计算改进的加权Jaccard相似度，结合词频权重
    """
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    if not words1 or not words2:
        return 0.0

    counter1 = Counter(words1)
    counter2 = Counter(words2)

    # 计算所有词的并集
    all_words = set(counter1.keys()).union(set(counter2.keys()))

    # 计算最小权重和与最大权重和
    min_sum = 0
    max_sum = 0

    for word in all_words:
        count1 = counter1.get(word, 0)
        count2 = counter2.get(word, 0)
        min_sum += min(count1, count2)
        max_sum += max(count1, count2)

    return min_sum / max_sum if max_sum != 0 else 0.0


def word_overlap_similarity(text1: str, text2: str) -> float:
    """
    计算基于词汇重叠的相似度
    """
    words1 = set(preprocess_text(text1))
    words2 = set(preprocess_text(text2))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    min_length = min(len(words1), len(words2))

    return intersection / min_length if min_length != 0 else 0.0


def calculate_combined_similarity(text1: str, text2: str) -> float:
    """
    综合多种相似度算法计算最终相似度
    """
    # 计算各种相似度
    cosine_sim = cosine_similarity_numpy(text1, text2)
    jaccard_sim = jaccard_similarity(text1, text2)
    weighted_jaccard_sim = weighted_jaccard_similarity(text1, text2)
    overlap_sim = word_overlap_similarity(text1, text2)

    # 动态权重调整（根据文本长度）
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)
    text_length = min(len(words1), len(words2))

    # 短文本更适合Jaccard，长文本更适合余弦相似度
    if text_length < 50:
        weights = [0.2, 0.3, 0.3, 0.2]  # 偏向Jaccard
    else:
        weights = [0.4, 0.2, 0.2, 0.2]  # 偏向余弦相似度

    # 计算加权平均
    combined_sim = (cosine_sim * weights[0] +
                    jaccard_sim * weights[1] +
                    weighted_jaccard_sim * weights[2] +
                    overlap_sim * weights[3])

    return combined_sim


def main():
    """
    主函数，处理命令行参数并执行查重
    """
    parser = argparse.ArgumentParser(description='论文查重系统')
    parser.add_argument('original_file', help='原文文件路径')
    parser.add_argument('copied_file', help='抄袭版论文文件路径')
    parser.add_argument('output_file', help='输出结果文件路径')
    parser.add_argument('--algorithm', '-a', choices=['cosine', 'jaccard', 'weighted_jaccard', 'overlap', 'combined'],
                        default='combined', help='选择相似度算法 (默认: combined)')

    args = parser.parse_args()

    try:
        # 读取文件内容
        original_text = read_file(args.original_file)
        copied_text = read_file(args.copied_file)

        # 根据选择的算法计算相似度
        if args.algorithm == 'cosine':
            similarity = cosine_similarity_numpy(original_text, copied_text)
        elif args.algorithm == 'jaccard':
            similarity = jaccard_similarity(original_text, copied_text)
        elif args.algorithm == 'weighted_jaccard':
            similarity = weighted_jaccard_similarity(original_text, copied_text)
        elif args.algorithm == 'overlap':
            similarity = word_overlap_similarity(original_text, copied_text)
        else:  # combined
            similarity = calculate_combined_similarity(original_text, copied_text)

        # 格式化结果，保留两位小数
        result = f"{similarity:.2f}"

        # 写入输出文件
        write_file(args.output_file, result)

        print(f"查重完成，使用算法: {args.algorithm}, 重复率为: {result}")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
