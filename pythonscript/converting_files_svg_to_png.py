import os
import cairosvg

# Define the directory containing SVG files
input_directory = '/Users/elifnazduman/Desktop/E.D ÖZEL/python/uml-parser/data_models/53_LibrePCB'
output_directory = '/Users/elifnazduman/Desktop/E.D ÖZEL/python/uml-parser/data_models/53_LibrePCB'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    if filename.lower().endswith('.svg'):
        svg_path = os.path.join(input_directory, filename)
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_png_path = os.path.join(output_directory, output_filename)

        # Convert SVG to PNG
        cairosvg.svg2png(url=svg_path, write_to=output_png_path)
        print(f"Converted {svg_path} to {output_png_path}")

print("All SVG files have been converted to PNG.")
