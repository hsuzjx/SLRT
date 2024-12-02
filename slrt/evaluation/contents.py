from slrt.evaluation.utils import clean_phoenix_2014, clean_phoenix_2014_trans

DatasetCleanFunctionDict = {
    "phoenix2014": clean_phoenix_2014,
    "phoenix2014-keypoint": clean_phoenix_2014,
    "phoenix2014T": clean_phoenix_2014_trans,
    "phoenix2014T-keypoint": clean_phoenix_2014_trans,
}
