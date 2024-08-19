import unittest
from unittest.mock import patch, mock_open
from src.evaluation.utils import modify_phoenix2014_output, read_file, write_file


class TestModifyPhoenix2014Output(unittest.TestCase):

    def setUp(self):
        self.input_content = [
            "01April_2010_Thursday_heute_default-1 1 0.02 0.03 WETTER",
            "01April_2010_Thursday_heute_default-1 2 0.03 0.04 SONNE",
            "01April_2010_Thursday_heute_default-1 3 0.04 0.05 cl-DIE",
            "01April_2010_Thursday_heute_default-1 4 0.05 0.06 loc-S0NNE",
            "01April_2010_Thursday_heute_default-1 9 0.10 0.11 __PU__",
            "01April_2010_Thursday_heute_default-1 10 0.11 0.12 __LEFTHAND__",
            "01April_2010_Thursday_heute_default-1 11 0.12 0.13 HABEN2",
            "01April_2010_Thursday_heute_default-1 12 0.13 0.14 WIE AUSSEHEN",
            "01April_2010_Thursday_heute_default-1 13 0.14 0.15 ZEIGEN ",
            "01April_2010_Thursday_heute_default-1 14 0.15 0.16 ZEIGEN$",
            "01April_2010_Thursday_heute_default-2 1 0.16 0.17 DAS",
            "01April_2010_Thursday_heute_default-2 2 0.17 0.18 IST",
            "01April_2010_Thursday_heute_default-1 5 0.06 0.07 qu-WIE",
            "01April_2010_Thursday_heute_default-1 6 0.07 0.08 poss-DER",
            "01April_2010_Thursday_heute_default-1 7 0.08 0.09 lh-BUCH",
            "01April_2010_Thursday_heute_default-1 8 0.09 0.10 __EMOTION__",
            "01April_2010_Thursday_heute_default-2 3 0.18 0.19 EIN",
            "01April_2010_Thursday_heute_default-2 4 0.19 0.20 TEST"
        ]
        self.expected_output_content = [
            "01April_2010_Thursday_heute_default-1 1 0.02 0.03 WETTER",
            "01April_2010_Thursday_heute_default-1 2 0.03 0.04 SONNE",
            "01April_2010_Thursday_heute_default-1 3 0.04 0.05 DIE",
            "01April_2010_Thursday_heute_default-1 4 0.05 0.06 SONNE",
            "01April_2010_Thursday_heute_default-1 5 0.06 0.07 WIE",
            "01April_2010_Thursday_heute_default-1 6 0.07 0.08 DER",
            "01April_2010_Thursday_heute_default-1 7 0.08 0.09 BUCH",
            "01April_2010_Thursday_heute_default-1 11 0.12 0.13 HABEN",
            "01April_2010_Thursday_heute_default-1 12 0.13 0.14 WIE-AUSSEHEN",
            "01April_2010_Thursday_heute_default-1 13 0.14 0.15 ZEIGEN-BILDSCHIRM",
            "01April_2010_Thursday_heute_default-1 14 0.15 0.16 ZEIGEN-BILDSCHIRM",
            "01April_2010_Thursday_heute_default-2 1 0.16 0.17 DAS",
            "01April_2010_Thursday_heute_default-2 2 0.17 0.18 IST",
            "01April_2010_Thursday_heute_default-2 3 0.18 0.19 EIN",
            "01April_2010_Thursday_heute_default-2 4 0.19 0.20 TEST"
        ]

    def test_modify_phoenix2014_output(self):
        # Call the function to test
        modify_phoenix2014_output('./test.ctm', './output.ctm')



if __name__ == '__main__':
    unittest.main()
