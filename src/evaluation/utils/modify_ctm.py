import re

from .read_file import read_file
from .write_file import write_file


def modify_phoenix2014_output(input_file, output_file):
    """
    Modify the output of the phoenix2014 dataset to be compatible with the evaluation script
    :param input_file: The input file path
    :param output_file: The output file path
    :return: None
    """
    # TODO: 必要的单元测试
    # Read the input file
    lines = read_file(input_file)

    # Apply sed commands
    processed_lines = []
    for line in lines:
        # Remove unwanted prefixes and special tokens
        line = re.sub(r'loc-|cl-|qu-|poss-|lh-|__EMOTION__|__PU__|__LEFTHAND__', '', line)
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
                      not re.search(r'__LEFTHAND__|__EPENTHESIS__|__EMOTION__', line)]

    # Process data with awk logic
    processed_data = []
    last_id = ""
    cnt = {}
    for line in filtered_lines:
        parts = line.split()
        if len(parts) >= 5:
            current_id = parts[0]
            # Check if the current ID is different from the last and if the last ID has less than 1 count, then add an empty marker
            if last_id != current_id and cnt.get(last_id, 0) < 1 and last_id:
                processed_data.append(f"{last_row} [EMPTY]")
            # Count occurrences of the current ID and add the line to the processed data
            if parts[4]:
                cnt[current_id] = cnt.get(current_id, 0) + 1
                processed_data.append(line)
            last_id = current_id
            last_row = line
    # Check for the last ID to add an empty marker if necessary
    if last_id and cnt.get(last_id, 0) < 1:
        processed_data.append(f"{last_row} [EMPTY]")

    # Sort the final output by specific criteria
    sorted_data = sorted(processed_data, key=lambda x: (x.split()[0], x.split()[2]))

    # Write to output file
    write_file(output_file, sorted_data)
