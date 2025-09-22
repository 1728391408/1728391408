# -*- coding: utf-8 -*-
"""
论文查重系统主模块
基于余弦相似度和多种文本相似度算法的论文查重工具
支持中文文本处理和多编码格式文件读取
"""

import math
from collections import Counter
import argparse
import sys
import os
from typing import List, Set, Dict, Tuple, Optional

# 第三方库导入
import jieba


def read_file(file_path: str) -> str:
    """
    读取文件内容，尝试多种编码格式

    Args:
        file_path: 文件路径

    Returns:
        文件内容字符串

    Raises:
        FileNotFoundError: 文件不存在时抛出
        IOError: 文件读取错误时抛出
        UnicodeDecodeError: 编码错误时抛出
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not os.path.isfile(file_path):
        raise IOError(f"路径不是文件: {file_path}")

    encodings = ['utf-8', 'gbk', 'gb2312', 'big5']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                print(f"成功读取文件: {file_path} (编码: {encoding})")
                return content
        except UnicodeDecodeError:
            continue
        except PermissionError as exc:
            raise PermissionError(f"没有权限读取文件: {file_path}") from exc
        except Exception as exc:
            # 如果是最后一次尝试，抛出异常
            if encoding == encodings[-1]:
                raise IOError(f"无法读取文件: {file_path}") from exc
            continue

    raise UnicodeDecodeError(f"无法解码文件: {file_path}")


def write_file(file_path: str, content: str) -> None:
    """
    写入文件内容

    Args:
        file_path: 文件路径
        content: 要写入的内容

    Raises:
        IOError: 文件写入错误时抛出
    """
    try:
        # 获取文件所在目录
        directory = os.path.dirname(file_path)

        # 如果目录不存在且不为空，则创建目录
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")

        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"文件已成功写入: {file_path}")

    except PermissionError as exc:
        raise PermissionError(f"没有权限写入文件: {file_path}") from exc
    except OSError as exc:
        raise OSError(f"操作系统错误，无法写入文件: {file_path}") from exc
    except Exception as exc:
        raise IOError(f"无法写入文件: {file_path}") from exc


def preprocess_text(text: str) -> List[str]:
    """
    文本预处理：分词并过滤停用词

    Args:
        text: 输入文本

    Returns:
        处理后的词汇列表
    """
    # 停用词集合
    stopwords = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这'
    }

    # 使用jieba分词
    words = jieba.cut(text)

    # 过滤停用词和空白字符
    return [word for word in words if word.strip() and word not in stopwords]


def cosine_similarity_numpy(text1: str, text2: str) -> float:
    """
    使用NumPy优化计算两段文本的余弦相似度

    Args:
        text1: 第一段文本
        text2: 第二段文本

    Returns:
        余弦相似度值 (0-1之间)
    """
    # 延迟导入NumPy以避免不必要的依赖
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel,reimported
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
    vector1 = np.array([counter1.get(word, 0) for word in all_words],
                       dtype=np.float32)
    vector2 = np.array([counter2.get(word, 0) for word in all_words],
                       dtype=np.float32)

    # 计算点积
    dot_product = np.dot(vector1, vector2)

    # 计算模长
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # 避免除以零
    if magnitude1 * magnitude2 == 0:
        return 0.0

    # 计算余弦相似度
    return float(dot_product / (magnitude1 * magnitude2))


def cosine_similarity_fallback(text1: str, text2: str) -> float:
    """
    回退方案：使用原生Python计算余弦相似度

    Args:
        text1: 第一段文本
        text2: 第二段文本

    Returns:
        余弦相似度值 (0-1之间)
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

    Args:
        text1: 第一段文本
        text2: 第二段文本

    Returns:
        Jaccard相似度值 (0-1之间)
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

    Args:
        text1: 第一段文本
        text2: 第二段文本

    Returns:
        加权Jaccard相似度值 (0-1之间)
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

    Args:
        text1: 第一段文本
        text2: 第二段文本

    Returns:
        词汇重叠相似度值 (0-1之间)
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

    Args:
        text1: 第一段文本
        text2: 第二段文本

    Returns:
        综合相似度值 (0-1之间)
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


def main() -> None:
    """
    主函数，处理命令行参数并执行查重
    """
    parser = argparse.ArgumentParser(description='论文查重系统')
    parser.add_argument('original_file', help='原文文件路径')
    parser.add_argument('copied_file', help='抄袭版论文文件路径')
    parser.add_argument('output_file', help='输出结果文件路径')
    parser.add_argument('--algorithm', '-a',
                        choices=['cosine', 'jaccard', 'weighted_jaccard',
                                 'overlap', 'combined'],
                        default='combined',
                        help='选择相似度算法 (默认: combined)')

    args = parser.parse_args()

    try:
        # 检查输入文件是否存在
        if not os.path.exists(args.original_file):
            raise FileNotFoundError(f"原文文件不存在: {args.original_file}")

        if not os.path.exists(args.copied_file):
            raise FileNotFoundError(f"抄袭文件不存在: {args.copied_file}")

        # 读取文件内容
        original_text = read_file(args.original_file)
        copied_text = read_file(args.copied_file)

        # 根据选择的算法计算相似度
        algorithm_functions = {
            'cosine': cosine_similarity_numpy,
            'jaccard': jaccard_similarity,
            'weighted_jaccard': weighted_jaccard_similarity,
            'overlap': word_overlap_similarity,
            'combined': calculate_combined_similarity
        }

        similarity_function = algorithm_functions[args.algorithm]
        similarity = similarity_function(original_text, copied_text)

        # 格式化结果，保留两位小数
        result = f"{similarity:.2f}"

        # 写入输出文件
        write_file(args.output_file, result)

        print(f"查重完成，使用算法: {args.algorithm}, 重复率为: {result}")

    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"程序执行错误: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()
