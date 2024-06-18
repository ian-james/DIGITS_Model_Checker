import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob

def remove_duplicate_columns(df):
    # Transpose the DataFrame, drop duplicate rows, then transpose back
    duplicated_columns = df.T.duplicated()
    # Select columns that are not duplicates
    df_no_duplicates = df.loc[:, ~duplicated_columns]
    return df_no_duplicates

def read_file(matching_file, sep='\t', sheet_name=0):

    get_extension = lambda file: file.split('.')[-1]
    extension = get_extension(matching_file)    

    matching_file
    if extension == 'csv':
        df = pd.read_csv(matching_file,sep=sep)
    elif extension in ['xlsx', 'xls']:
        df = pd.read_excel(matching_file, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    return df

def load_data(files, sep='\t', sheet_name=0):
    """Load and concatenate data from multiple CSV files."""
    df_list = []
    for file in files:
        try:
            df = read_file(file,sep,sheet_name)
            df_list.append(df)
        except ValueError as e:
            print(f"Error reading file: {file}, {e}")
            continue
        except FileNotFoundError as e:
            print(e)    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    if( len(df_list) == 0):
        print("No files read")
        return None

    concatenated_df = pd.concat(df_list,axis=1, ignore_index=False)
    concatenated_df = remove_duplicate_columns(concatenated_df)    
    return concatenated_df

def plot_data(df, output_dir):
    """Plot the max deviations for each column and save the plot as a PNG file."""
    plt.figure(figsize=(14, 8))

    # Aggregate max deviations by column
    df_agg = df.groupby('column')['max_deviation'].max().reset_index()

    plt.barh(df_agg['column'], df_agg['max_deviation'], color='skyblue')
    plt.xlabel('Max Deviation')
    plt.ylabel('Column')
    plt.title('Max Deviation for Each Column')
    plt.grid(True)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'max_deviation_summary.png')
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Process and plot max deviations from multiple CSV files.')
    parser.add_argument('-i', '--input_folder', type=str, default="./output/all_combined_fps_60_md_04/excel/", help='Input directory containing the CSV files.')
    parser.add_argument('-o', '--output_file', type=str, default="all_max_deviations.csv", help='Output directory to save the plot.')
    parser.add_argument('-e','--extension', type=str, default='xlsx', help='Extension of the input files (default: csv)')
    parser.add_argument('-n','--nfiles',  type=int, default=0, help='Number of files to process (default: 10)')
    parser.add_argument('-s','--sheet_name', type=str, default=0, help='Sheet name for Excel files (default: 0)')
    parser.add_argument('-t','--seperator', type=str, default='\t', help='Separator for CSV files (default: tab)')    

    args = vars(parser.parse_args())

    # Load all CSV files from the input directory
    if( not os.path.exists(args['input_folder'])):
        print(f"Input directory not found: {args['input_folder']}")
        return

    search_expresion = os.path.join(args['input_folder'], f"*.{args['extension']}")
    input_files = glob.glob(search_expresion)

    if( args['nfiles'] > 0):
        input_files = input_files[:args['nfiles']]

    if len(input_files) == 0:
        print(f"No files found in directory: {args['input_folder']}")
        return

    # Load and concatenate data
    df = load_data(input_files, sep=args['seperator'] ,sheet_name=args['sheet_name'])

    if(df is None):
        print("No data loaded")
        return

    # Plot the data
    # plot_data(df, args.output)
    # print(f"Plot saved to {args.output}")
    df.to_csv(os.path.join(args['input_folder'], args['output_file']), index=False)

if __name__ == '__main__':
    main()
