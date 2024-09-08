import re

from .read_file import read_file
from .write_file import write_file


def format_phoenix2014_output(input_file, output_file):
    """
    Modify the output of the phoenix2014 dataset to be compatible with the evaluation script
    :param input_file: The input file path
    :param output_file: The output file path
    :return: None
    """
    # Read the input file
    lines = read_file(input_file)

    # Apply sed commands
    processed_lines = []
    for line in lines:
        # Remove unwanted prefixes and special tokens
        line = re.sub(r'loc-|cl-|qu-|poss-|lh-', '', line)
        # Correct specific errors
        line = re.sub(r'S0NNE', 'SONNE', line)
        line = re.sub(r'HABEN2', 'HABEN', line)
        line = re.sub(r'WIE AUSSEHEN', 'WIE-AUSSEHEN', line)
        line = re.sub(r'ZEIGEN ', 'ZEIGEN-BILDSCHIRM ', line)
        line = re.sub(r'ZEIGEN$', 'ZEIGEN-BILDSCHIRM', line)
        # Adjust capitalization and spacing for compound words
        line = re.sub(r'^([A-Z]) ([A-Z][+ ])', r'\1+\2', line)
        line = re.sub(r'[ +]([A-Z]) ([A-Z]) ', r' \1+\2 ', line)
        line = re.sub(r'([ +][A-Z]) ([A-Z][ +])', r'\1+\2', line)
        line = re.sub(r'([ +]SCH) ([A-Z][ +])', r'\1+\2', line)
        line = re.sub(r'([ +]NN) ([A-Z][ +])', r'\1+\2', line)
        line = re.sub(r'([ +][A-Z]) (NN[ +])', r'\1+\2', line)
        line = re.sub(r'([ +][A-Z]) ([A-Z])$', r'\1+\2', line)
        # Remove unnecessary compound word components
        line = re.sub(r'([A-Z][A-Z])RAUM', r'\1', line)
        line = re.sub(r'-PLUSPLUS', '', line)

        # Apply perl command to remove duplicate words
        line = re.sub(r'(?<![a-zA-Z-])(\b[A-Z]+(?![a-zA-Z-])) \1(?![a-zA-Z-])', r'\1', line)

        # Remove trailing spaces
        line = line.rstrip()

        processed_lines.append(line)

    # Filter lines to remove specific unwanted tokens
    filtered_lines = [line for line in processed_lines if
                      not re.search(r'__LEFTHAND__|__EPENTHESIS__|__EMOTION__|__PU__', line)]

    # Write to output file
    write_file(output_file, filtered_lines)
