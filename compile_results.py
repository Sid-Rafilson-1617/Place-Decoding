import numpy as np
import pandas as pd
import os
from scipy.io import savemat

OB_dir = r"E:\place_decoding\data\bulb"
hippocampus_dir = r"E:\place_decoding\data\hipp"


def compile_full_session_data(OB_dir, hippocampus_dir):

    decoding_error_file = "decoding_err.npy"
    decoding_error_shuffle = "decoding_err_shuffle.npy"

    for region, region_name in zip([OB_dir, hippocampus_dir], ["OB", "hippocampus"]):

        # Create a DataFrame to store the results
        df = pd.DataFrame(columns=["Mouse", "Session", "DecodingError", "DecodingErrorShuffled"])

        # Loop through the directories and load the data
        mice = os.listdir(region)
        for mouse in mice:
            if not mouse.isnumeric():
                continue
            sessions = os.listdir(os.path.join(region, mouse))
            for session in sessions:
                session_path = os.path.join(region, mouse, session)
                if decoding_error_file in os.listdir(session_path):
                    decoding_error = np.load(os.path.join(session_path, decoding_error_file)).tolist()
                    decoding_error_shuffled = np.load(os.path.join(session_path, decoding_error_shuffle)).tolist()

                    # Append the current session's results to the DataFrame
                    current_results = {
                        "Mouse": mouse,
                        "Session": session,
                        "DecodingError": decoding_error,
                        "DecodingErrorShuffled": decoding_error_shuffled
                    }
                    df = df._append(current_results, ignore_index=True)

        # Save DataFrame to a MATLAB .mat file
        print(df)
        output_path = os.path.join(region, f'{region_name}_decoding_errors.mat')
        savemat(output_path, {"decoding_errors": df.to_dict("list")})

        # save as csv
        output_path = os.path.join(region, f'{region_name}_decoding_errors.csv')
        df.to_csv(output_path, index=False)

        print(f"Data saved to {output_path}")


def compile_floorflip_data(OB_dir):

    decoding_error_train_post_test_post_file = "decoding_err_train_post_test_post.npy"
    decoding_error_train_post_test_pre_file = "decoding_err_train_post_test_pre.npy"
    decoding_error_train_pre_test_post_file = "decoding_err_train_pre_test_post.npy"
    decoding_error_train_pre_test_pre_file = "decoding_err_train_pre_test_pre.npy"
    decoding_error_train_post_test_prerotated_file = "decoding_err_train_post_test_prerotated.npy"

    decoding_error_train_post_test_post_shuffle_file = "decoding_err_train_post_test_post_shuffle.npy"
    decoding_error_train_post_test_pre_shuffle_file = "decoding_err_train_post_test_pre_shuffle.npy"
    decoding_error_train_pre_test_post_shuffle_file = "decoding_err_train_pre_test_post_shuffle.npy"
    decoding_error_train_pre_test_pre_shuffle_file = "decoding_err_train_pre_test_pre_shuffle.npy"
    decoding_error_train_post_test_prerotated_shuffle_file = "decoding_err_train_post_test_prerotated_shuffle.npy"

    # Create a DataFrame to store the results
    df = pd.DataFrame(columns=["Mouse", "Session", "split", "DecodingError", "DecodingErrorShuffled"])

    # Loop through the directories and load the data
    mice = os.listdir(OB_dir)
    for mouse in mice:
        if not mouse.isnumeric():
            continue
        sessions = os.listdir(os.path.join(OB_dir, mouse))
        for session in sessions:
            session_path = os.path.join(OB_dir, mouse, session)
            if decoding_error_train_post_test_post_file in os.listdir(session_path):
                decoding_error_train_post_test_post = np.load(os.path.join(session_path, decoding_error_train_post_test_post_file)).tolist()
                decoding_error_train_post_test_pre = np.load(os.path.join(session_path, decoding_error_train_post_test_pre_file)).tolist()
                decoding_error_train_pre_test_post = np.load(os.path.join(session_path, decoding_error_train_pre_test_post_file)).tolist()
                decoding_error_train_pre_test_pre = np.load(os.path.join(session_path, decoding_error_train_pre_test_pre_file)).tolist()
                decoding_error_train_post_test_prerotated = np.load(os.path.join(session_path, decoding_error_train_post_test_prerotated_file)).tolist()

                decoding_error_train_post_test_post_shuffle = np.load(os.path.join(session_path, decoding_error_train_post_test_post_shuffle_file)).tolist()
                decoding_error_train_post_test_pre_shuffle = np.load(os.path.join(session_path, decoding_error_train_post_test_pre_shuffle_file)).tolist()
                decoding_error_train_pre_test_post_shuffle = np.load(os.path.join(session_path, decoding_error_train_pre_test_post_shuffle_file)).tolist()
                decoding_error_train_pre_test_pre_shuffle = np.load(os.path.join(session_path, decoding_error_train_pre_test_pre_shuffle_file)).tolist()
                decoding_error_train_post_test_prerotated_shuffle = np.load(os.path.join(session_path, decoding_error_train_post_test_prerotated_shuffle_file)).tolist()

                # Append the current session's results to the DataFrame
                current_results = {
                    "Mouse": mouse,
                    "Session": session,
                    "split": "postpost",
                    "DecodingError": decoding_error_train_post_test_post,
                    "DecodingErrorShuffled": decoding_error_train_post_test_post_shuffle
                }
                df = df._append(current_results, ignore_index=True)

                current_results = {
                    "Mouse": mouse,
                    "Session": session,
                    "split": "postpre",
                    "DecodingError": decoding_error_train_post_test_pre,
                    "DecodingErrorShuffled": decoding_error_train_post_test_pre_shuffle
                }
                df = df._append(current_results, ignore_index=True)

                current_results = {
                    "Mouse": mouse,
                    "Session": session,
                    "split": "prepost",
                    "DecodingError": decoding_error_train_pre_test_post,
                    "DecodingErrorShuffled": decoding_error_train_pre_test_post_shuffle
                }
                df = df._append(current_results, ignore_index=True)

                current_results = { 
                    "Mouse": mouse,
                    "Session": session,
                    "split": "prepre",
                    "DecodingError": decoding_error_train_pre_test_pre,
                    "DecodingErrorShuffled": decoding_error_train_pre_test_pre_shuffle
                }
                df = df._append(current_results, ignore_index=True)

                current_results = { 
                    "Mouse": mouse,
                    "Session": session,
                    "split": "postprerotated",
                    "DecodingError": decoding_error_train_post_test_prerotated,
                    "DecodingErrorShuffled": decoding_error_train_post_test_prerotated_shuffle
                }
                df = df._append(current_results, ignore_index=True)

    # Save DataFrame to a MATLAB .mat file
    print(df)
    output_path = os.path.join(OB_dir, 'floorflip_decoding_errors.mat')
    savemat(output_path, {"decoding_errors": df.to_dict("list")})

    # save as csv
    output_path = os.path.join(OB_dir, 'floorflip_decoding_errors.csv')
    df.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")




if __name__ == "__main__":
    compile_floorflip_data(OB_dir)
    compile_full_session_data(OB_dir, hippocampus_dir)

    
    