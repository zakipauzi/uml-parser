import os
# import cairosvg

def ensure_directory_exists(model_dir, repo_name):
    os.makedirs(model_dir + repo_name, exist_ok=True)


def run_conversion():

    # Iterate over all files in the input directory

    model_dir = 'data_models'

    for folder in os.listdir(model_dir):

        print(folder)

    #     if os.path.isdir(os.path.join(model_dir, folder)):
    #         ensure_directory_exists(model_dir, folder)

    #     for filename in os.listdir(input_directory):
    #         if filename.lower().endswith('.svg'):
    #             svg_path = os.path.join(input_directory, filename)
    #             output_filename = os.path.splitext(filename)[0] + '.png'
    #             output_png_path = os.path.join(output_directory, output_filename)

    #             # Convert SVG to PNG
    #             cairosvg.svg2png(url=svg_path, write_to=output_png_path)
    #             print(f"Converted {svg_path} to {output_png_path}")

    return "All SVG files have been converted to PNG."
