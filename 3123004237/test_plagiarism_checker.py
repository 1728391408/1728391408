#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
论文查重系统测试用例
包含至少10个测试用例，覆盖各种边界情况和功能
"""

import unittest
import os
import tempfile
import sys
from unittest.mock import patch, mock_open
from io import StringIO

# 导入待测试的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    read_file, write_file, preprocess_text,
    cosine_similarity_numpy, jaccard_similarity,
    weighted_jaccard_similarity, word_overlap_similarity,
    calculate_combined_similarity, main
)


class TestPlagiarismChecker(unittest.TestCase):
    """论文查重系统测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()

        # 创建测试文件
        self.original_file = os.path.join(self.test_dir, "original.txt")
        self.copied_file = os.path.join(self.test_dir, "copied.txt")
        self.output_file = os.path.join(self.test_dir, "output.txt")

        # 测试文本
        self.text1 = "这是一段测试文本，用于测试文本相似度计算算法的准确性。"
        self.text2 = "这是一段完全不同的文本，内容主题完全不同。"
        self.text3 = "这是一段测试文本，用于测试文本相似度计算算法的准确性。但增加了一些额外的内容。"

        # 写入测试文件
        with open(self.original_file, 'w', encoding='utf-8') as f:
            f.write(self.text1)
        with open(self.copied_file, 'w', encoding='utf-8') as f:
            f.write(self.text3)

    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_1_identical_texts(self):
        """测试1: 完全相同的文本"""
        cosine_sim = cosine_similarity_numpy(self.text1, self.text1)
        jaccard_sim = jaccard_similarity(self.text1, self.text1)
        weighted_jaccard_sim = weighted_jaccard_similarity(self.text1, self.text1)
        overlap_sim = word_overlap_similarity(self.text1, self.text1)
        combined_sim = calculate_combined_similarity(self.text1, self.text1)

        # 所有相似度应该接近1.0
        self.assertAlmostEqual(cosine_sim, 1.0, places=2)
        self.assertAlmostEqual(jaccard_sim, 1.0, places=2)
        self.assertAlmostEqual(weighted_jaccard_sim, 1.0, places=2)
        self.assertAlmostEqual(overlap_sim, 1.0, places=2)
        self.assertAlmostEqual(combined_sim, 1.0, places=2)

    def test_2_completely_different_texts(self):
        """测试2: 完全不同的文本"""
        cosine_sim = cosine_similarity_numpy(self.text1, self.text2)
        jaccard_sim = jaccard_similarity(self.text1, self.text2)
        weighted_jaccard_sim = weighted_jaccard_similarity(self.text1, self.text2)
        overlap_sim = word_overlap_similarity(self.text1, self.text2)
        combined_sim = calculate_combined_similarity(self.text1, self.text2)

        # 所有相似度应该接近0.0
        self.assertLess(cosine_sim, 0.3)
        self.assertLess(jaccard_sim, 0.3)
        self.assertLess(weighted_jaccard_sim, 0.3)
        self.assertLess(overlap_sim, 0.3)
        self.assertLess(combined_sim, 0.3)

    def test_3_partially_similar_texts(self):
        """测试3: 部分相似的文本"""
        cosine_sim = cosine_similarity_numpy(self.text1, self.text3)
        jaccard_sim = jaccard_similarity(self.text1, self.text3)
        weighted_jaccard_sim = weighted_jaccard_similarity(self.text1, self.text3)
        overlap_sim = word_overlap_similarity(self.text1, self.text3)
        combined_sim = calculate_combined_similarity(self.text1, self.text3)

        # 相似度应该在0.5-0.9之间
        self.assertGreater(cosine_sim, 0.5)
        self.assertLess(cosine_sim, 0.9)
        self.assertGreater(jaccard_sim, 0.5)
        self.assertLess(jaccard_sim, 0.9)
        self.assertGreater(weighted_jaccard_sim, 0.5)
        self.assertLess(weighted_jaccard_sim, 0.9)
        self.assertGreater(overlap_sim, 0.5)
        self.assertLess(overlap_sim, 0.9)
        self.assertGreater(combined_sim, 0.5)
        self.assertLess(combined_sim, 0.9)

    def test_4_empty_text(self):
        """测试4: 空文本处理"""
        empty_text = ""

        cosine_sim = cosine_similarity_numpy(self.text1, empty_text)
        jaccard_sim = jaccard_similarity(self.text1, empty_text)
        weighted_jaccard_sim = weighted_jaccard_similarity(self.text1, empty_text)
        overlap_sim = word_overlap_similarity(self.text1, empty_text)
        combined_sim = calculate_combined_similarity(self.text1, empty_text)

        # 所有相似度应该为0.0
        self.assertEqual(cosine_sim, 0.0)
        self.assertEqual(jaccard_sim, 0.0)
        self.assertEqual(weighted_jaccard_sim, 0.0)
        self.assertEqual(overlap_sim, 0.0)
        self.assertEqual(combined_sim, 0.0)

    def test_5_both_empty_texts(self):
        """测试5: 两个空文本"""
        empty_text = ""

        cosine_sim = cosine_similarity_numpy(empty_text, empty_text)
        jaccard_sim = jaccard_similarity(empty_text, empty_text)
        weighted_jaccard_sim = weighted_jaccard_similarity(empty_text, empty_text)
        overlap_sim = word_overlap_similarity(empty_text, empty_text)
        combined_sim = calculate_combined_similarity(empty_text, empty_text)

        # 所有相似度应该为0.0
        self.assertEqual(cosine_sim, 0.0)
        self.assertEqual(jaccard_sim, 0.0)
        self.assertEqual(weighted_jaccard_sim, 0.0)
        self.assertEqual(overlap_sim, 0.0)
        self.assertEqual(combined_sim, 0.0)

    def test_6_file_operations(self):
        """测试6: 文件读写功能"""
        test_content = "这是一段测试内容，用于测试文件读写功能。"

        # 测试写入文件
        write_file(self.output_file, test_content)

        # 测试读取文件
        read_content = read_file(self.output_file)

        # 验证内容一致
        self.assertEqual(test_content, read_content)

    def test_7_preprocess_text(self):
        """测试7: 文本预处理功能"""
        text = "这是一段的测试文本，包含了停用词和有效词。"
        processed = preprocess_text(text)

        # 检查停用词是否被过滤
        self.assertNotIn("的", processed)
        self.assertNotIn("了", processed)
        self.assertNotIn("和", processed)

        # 检查有效词是否保留
        self.assertIn("测试", processed)
        self.assertIn("文本", processed)
        self.assertIn("有效", processed)
        self.assertIn("词", processed)

    def test_8_different_encodings(self):
        """测试8: 不同编码格式的文件读取"""
        # 创建GBK编码的文件
        gbk_file = os.path.join(self.test_dir, "gbk_test.txt")
        gbk_content = "这是一段GBK编码的测试文本。"

        with open(gbk_file, 'w', encoding='gbk') as f:
            f.write(gbk_content)

        # 测试读取GBK编码的文件
        read_content = read_file(gbk_file)
        self.assertEqual(gbk_content, read_content)

    def test_9_nonexistent_file(self):
        """测试9: 文件不存在时的错误处理"""
        nonexistent_file = os.path.join(self.test_dir, "nonexistent.txt")

        # 测试读取不存在的文件
        with self.assertRaises(Exception) as context:
            read_file(nonexistent_file)

        self.assertIn("文件不存在", str(context.exception))

    def test_10_main_function(self):
        """测试10: 主函数功能"""
        # 使用patch模拟命令行参数
        test_args = [
            "plagiarism_checker.py",
            self.original_file,
            self.copied_file,
            self.output_file,
            "--algorithm",
            "cosine"
        ]

        with patch.object(sys, 'argv', test_args):
            # 捕获标准输出
            with patch('sys.stdout', new=StringIO()) as fake_output:
                main()
                output = fake_output.getvalue().strip()

                # 检查输出是否包含预期内容
                self.assertIn("查重完成", output)
                self.assertIn("重复率为", output)

        # 检查输出文件是否创建并包含合理的结果
        self.assertTrue(os.path.exists(self.output_file))

        with open(self.output_file, 'r', encoding='utf-8') as f:
            result = f.read().strip()

            # 检查结果是否为0.00到1.00之间的浮点数
            similarity = float(result)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)

    def test_11_short_texts(self):
        """测试11: 短文本处理"""
        short_text1 = "人工智能"
        short_text2 = "人工智障"

        cosine_sim = cosine_similarity_numpy(short_text1, short_text2)
        jaccard_sim = jaccard_similarity(short_text1, short_text2)

        # 短文本应该有一定的相似度但不是完全相似
        self.assertGreater(cosine_sim, 0.0)
        self.assertLess(cosine_sim, 1.0)
        self.assertGreater(jaccard_sim, 0.0)
        self.assertLess(jaccard_sim, 1.0)

    def test_12_long_texts(self):
        """测试12: 长文本处理"""
        # 生成长文本
        base_text = "这是一段基础文本，用于生成长文本测试。"
        long_text1 = base_text * 100
        long_text2 = base_text * 80 + "这是额外添加的内容，用于测试长文本相似度。" * 20

        cosine_sim = cosine_similarity_numpy(long_text1, long_text2)
        combined_sim = calculate_combined_similarity(long_text1, long_text2)

        # 长文本应该有较高的相似度
        self.assertGreater(cosine_sim, 0.7)
        self.assertGreater(combined_sim, 0.7)

    def test_13_stopword_handling(self):
        """测试13: 停用词处理"""
        text_with_stopwords = "的了一是和我不就人有都一个上也很有到说要"
        text_without_stopwords = "测试文本内容有效词"

        # 预处理应该过滤掉停用词
        processed1 = preprocess_text(text_with_stopwords)
        processed2 = preprocess_text(text_without_stopwords)

        # 第一个文本应该被过滤为空或很少的词
        self.assertLessEqual(len(processed1), 2)

        # 第二个文本应该保留大部分词
        self.assertGreaterEqual(len(processed2), 3)

        # 计算相似度
        similarity = cosine_similarity_numpy(text_with_stopwords, text_without_stopwords)

        # 相似度应该很低
        self.assertLess(similarity, 0.3)

    def test_14_all_algorithms(self):
        """测试14: 所有算法的一致性"""
        # 测试所有算法对同一对文本的结果
        algorithms = [
            ('cosine', cosine_similarity_numpy),
            ('jaccard', jaccard_similarity),
            ('weighted_jaccard', weighted_jaccard_similarity),
            ('overlap', word_overlap_similarity),
            ('combined', calculate_combined_similarity)
        ]

        results = {}
        for name, func in algorithms:
            results[name] = func(self.text1, self.text3)

        # 所有算法的结果应该在合理范围内
        for name, result in results.items():
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

            # 对于这对文本，所有算法的结果应该大于0.5
            self.assertGreater(result, 0.5)

    def test_15_command_line_interface(self):
        """测试15: 命令行接口测试"""
        # 测试不同的算法选项
        algorithms = ['cosine', 'jaccard', 'weighted_jaccard', 'overlap', 'combined']

        for algorithm in algorithms:
            test_args = [
                "plagiarism_checker.py",
                self.original_file,
                self.copied_file,
                self.output_file,
                "--algorithm",
                algorithm
            ]

            with patch.object(sys, 'argv', test_args):
                # 捕获标准输出
                with patch('sys.stdout', new=StringIO()) as fake_output:
                    main()
                    output = fake_output.getvalue().strip()

                    # 检查输出是否包含算法名称
                    self.assertIn(algorithm, output)
                    self.assertIn("重复率为", output)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
