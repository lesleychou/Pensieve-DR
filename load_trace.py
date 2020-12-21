'''
Loads trace training data.
'''

import os

def load_trace(cooked_trace_folder):
    # print("Loading traces from " + cooked_trace_folder)
    # cooked_files = os.listdir(cooked_trace_folder)
    # print("Found " + str(len(cooked_files)) + " trace files.")
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for subdir ,dirs ,files in os.walk( cooked_trace_folder ):
        for file in files:
            # print os.path.join(subdir, file)
            file_path = subdir + os.sep + file
            val_folder_name = os.path.basename( os.path.normpath( subdir ) )
            #print( val_folder_name, "-----")
            cooked_time = []
            cooked_bw = []
            # print file_path
            with open(file_path, 'rb') as phile:
                for line in phile:
                    parse = line.split()
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
            all_cooked_time.append(cooked_time)
            all_cooked_bw.append(cooked_bw)
            all_file_names.append(val_folder_name + '_' + file)

    return all_cooked_time, all_cooked_bw, all_file_names
