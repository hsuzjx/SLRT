import unittest
from unittest.mock import patch, MagicMock
from src.evaluation.wer_calculation import evaluate

# TODO: 添加测试用例
class TestEvaluate(unittest.TestCase):

    @patch('src.evaluation.wer_calculation.os')
    @patch('src.evaluation.wer_calculation.subprocess')
    @patch('src.evaluation.wer_calculation.merge_ctm_stm')
    def test_evaluate(self, mock_merge_ctm_stm, mock_subprocess, mock_os):
        # 设置模拟值
        mock_os.path.abspath.side_effect = lambda x: x
        mock_subprocess.run.return_value = MagicMock(stdout="Some output", stderr="")

        # 定义测试用的文件路径和目录
        file_save_path = "./test_dir"
        groundtruth_file = "./test_groundtruth.stm"
        ctm_file = "./test_output.ctm"
        evaluate_dir = "./evaluation_scripts"
        sclite_path = "./sclite_bin/sclite"

        # 调用待测试的函数
        wer = evaluate(file_save_path, groundtruth_file, ctm_file, evaluate_dir, sclite_path)

        # 验证函数调用和返回值
        mock_merge_ctm_stm.assert_called_once()
        mock_subprocess.run.assert_called()
        self.assertIsNone(wer)  # 因为函数没有返回值，所以预期是 None

if __name__ == '__main__':
    unittest.main()
