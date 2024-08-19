import os
from datetime import datetime

from .utils import merge_ctm_stm, sort_ctm
# Import utility functions for modifying, merging, and sorting CTM files
from .utils import modify_phoenix2014_output


def process_phoenix2014_output(file, ground_truth_file, processed_file, remove_tmp_file=True):
    """
    Preprocesses the output of the Phoenix 2014 dataset.

    This function modifies, merges, and sorts the CTM files to ensure the output meets the required format standards.
    
    Parameters:
    - file: The path to the original CTM file.
    - ground_truth_file: The path to the ground truth STM file.
    - processed_file: The path to the final processed CTM file.
    - remove_tmp_file: Whether to delete temporary files after processing.
    """
    # Get the directory and base name of the file for subsequent file operations
    file_save_dir = os.path.dirname(file)
    base_name = os.path.basename(file)

    # Prepare names for processed and merged CTM files
    modified_ctm_file = os.path.join(file_save_dir, f"tmp1.modified.{base_name}")
    merged_ctm_file = os.path.join(file_save_dir, f"tmp2.merged.{base_name}")

    # Output the start time of the preprocessing
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Processing CTM file...")

    # Modify the original CTM file to correct the format
    modify_phoenix2014_output(file, modified_ctm_file)

    # Merge CTM and STM files to ensure completeness of the transcription
    merge_ctm_stm(modified_ctm_file, ground_truth_file, merged_ctm_file)

    # Sort the merged CTM file to facilitate subsequent processing
    sort_ctm(merged_ctm_file, processed_file)

    # Optionally delete temporary files
    if remove_tmp_file:
        os.remove(modified_ctm_file)
        os.remove(merged_ctm_file)

    # Output completion time and processed file path
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Processing CTM file done. Output to {processed_file}")
