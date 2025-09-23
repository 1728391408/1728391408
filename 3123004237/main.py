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
from typing import List, Dict, Any

# 第三方库导入
import jieba


def read_file(file_path: str) -> str:
    """
    读取文件内容，尝试多种编码格式
    """
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
        except OSError as exc:
            if encoding == encodings[-1]:
                raise IOError(f"无法读取文件: {file_path}") from exc
            continue

    raise UnicodeDecodeError("utf-8", b"", 0, 0, f"无法解码文件: {file_path}")


def write_file(file_path: str, content: str) -> None:
    """
    写入文件内容
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")

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
    """
    stopwords = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
        '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
        '没有', '看', '好', '自己', '这'
    }

    words = jieba.cut(text)
    return [word for word in words if word.strip() and word not in stopwords]


def get_text_representation(text: str) -> Dict[str, Any]:
    """
    获取文本的多种表示形式，避免重复分词
    """
    words = preprocess_text(text)
    counter = Counter(words)
    word_set = set(words)

    return {
        'words': words,  # 原始词列表（保持顺序和重复）
        'counter': counter,  # 词频计数器
        'word_set': word_set,  # 词集合（去重）
        'length': len(words)  # 文本长度
    }


def cosine_similarity_numpy(text1: str, text2: str) -> float:
    """
    使用NumPy优化计算两段文本的余弦相似度
    """
    # 获取文本表示
    rep1 = get_text_representation(text1)
    rep2 = get_text_representation(text2)

    if not rep1['words'] or not rep2['words']:
        return 0.0

    # 延迟导入NumPy
    try:
        import numpy as np
    except ImportError:
        print("警告: NumPy未安装，使用原生Python计算余弦相似度")
        return cosine_similarity_fallback_with_reps(rep1, rep2)

    # 获取所有词汇的并集
    all_words = list(set(rep1['counter'].keys()).union(set(rep2['counter'].keys())))

    # 创建词频向量
    vector1 = np.array([rep1['counter'].get(word, 0) for word in all_words], dtype=np.float32)
    vector2 = np.array([rep2['counter'].get(word, 0) for word in all_words], dtype=np.float32)

    # 计算余弦相似度
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 * magnitude2 == 0:
        return 0.0

    return float(dot_product / (magnitude1 * magnitude2))


def cosine_similarity_fallback(text1: str, text2: str) -> float:
    """
    回退方案：使用原生Python计算余弦相似度
    """
    rep1 = get_text_representation(text1)
    rep2 = get_text_representation(text2)
    return cosine_similarity_fallback_with_reps(rep1, rep2)


def cosine_similarity_fallback_with_reps(rep1: Dict[str, Any], rep2: Dict[str, Any]) -> float:
    """
    使用原生Python计算余弦相似度（基于文本表示）
    """
    if not rep1['words'] or not rep2['words']:
        return 0.0

    all_words = set(rep1['counter'].keys()).union(set(rep2['counter'].keys()))

    vector1 = [rep1['counter'].get(word, 0) for word in all_words]
    vector2 = [rep2['counter'].get(word, 0) for word in all_words]

    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(v * v for v in vector1))
    magnitude2 = math.sqrt(sum(v * v for v in vector2))

    if magnitude1 * magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    计算两段文本的Jaccard相似度
    """
    rep1 = get_text_representation(text1)
    rep2 = get_text_representation(text2)
    return jaccard_similarity_with_reps(rep1, rep2)


def jaccard_similarity_with_reps(rep1: Dict[str, Any], rep2: Dict[str, Any]) -> float:
    """
    基于文本表示计算Jaccard相似度
    """
    if not rep1['word_set'] or not rep2['word_set']:
        return 0.0

    intersection = len(rep1['word_set'].intersection(rep2['word_set']))
    union = len(rep1['word_set'].union(rep2['word_set']))

    return intersection / union if union != 0 else 0.0


def weighted_jaccard_similarity(text1: str, text2: str) -> float:
    """
    计算改进的加权Jaccard相似度
    """
    rep1 = get_text_representation(text1)
    rep2 = get_text_representation(text2)
    return weighted_jaccard_similarity_with_reps(rep1, rep2)


def weighted_jaccard_similarity_with_reps(rep1: Dict[str, Any], rep2: Dict[str, Any]) -> float:
    """
    基于文本表示计算加权Jaccard相似度
    """
    if not rep1['counter'] or not rep2['counter']:
        return 0.0

    all_words = set(rep1['counter'].keys()).union(set(rep2['counter'].keys()))

    min_sum = 0
    max_sum = 0

    for word in all_words:
        count1 = rep1['counter'].get(word, 0)
        count2 = rep2['counter'].get(word, 0)
        min_sum += min(count1, count2)
        max_sum += max(count1, count2)

    return min_sum / max_sum if max_sum != 0 else 0.0


def word_overlap_similarity(text1: str, text2: str) -> float:
    """
    计算基于词汇重叠的相似度
    """
    rep1 = get_text_representation(text1)
    rep2 = get_text_representation(text2)
    return word_overlap_similarity_with_reps(rep1, rep2)


def word_overlap_similarity_with_reps(rep1: Dict[str, Any], rep2: Dict[str, Any]) -> float:
    """
    基于文本表示计算词汇重叠相似度
    """
    if not rep1['word_set'] or not rep2['word_set']:
        return 0.0

    intersection = len(rep1['word_set'].intersection(rep2['word_set']))
    min_length = min(len(rep1['word_set']), len(rep2['word_set']))

    return intersection / min_length if min_length != 0 else 0.0


def calculate_combined_similarity(text1: str, text2: str) -> float:
    """
    综合多种相似度算法计算最终相似度
    """
    # 获取文本表示
    rep1 = get_text_representation(text1)
    rep2 = get_text_representation(text2)

    if not rep1['words'] or not rep2['words']:
        return 0.0

    # 使用原有的计算方式，但基于预处理的文本表示
    cosine_sim = cosine_similarity_fallback_with_reps(rep1, rep2)
    jaccard_sim = jaccard_similarity_with_reps(rep1, rep2)
    weighted_jaccard_sim = weighted_jaccard_similarity_with_reps(rep1, rep2)
    overlap_sim = word_overlap_similarity_with_reps(rep1, rep2)

    # 使用原有的权重分配策略
    text_length = min(rep1['length'], rep2['length'])

    # 保持原有的权重分配
    if text_length < 50:
        weights = [0.2, 0.3, 0.3, 0.2]
    else:
        weights = [0.4, 0.2, 0.2, 0.2]

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
    except Exception as exc:
        print(f"程序执行错误: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()
