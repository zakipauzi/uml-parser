import os
import imageio


def run_gif_conversion(input_directory, output_directory=None):

    # If output_directory is not provided, use input_directory
    if output_directory is None:
        output_directory = input_directory

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.gif'):
            gif_path = os.path.join(input_directory, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_png_path = os.path.join(output_directory, output_filename)

            # Convert GIF to PNG
            gif = imageio.mimread(gif_path)
            imageio.mimsave(output_png_path, gif, format='PNG')
            print(f"Converted {gif_path} to {output_png_path}")

    print("All GIF files have been converted to PNG.")


# Example usage
run_gif_conversion('data_models/18_google-api-python-client')
