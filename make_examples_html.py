import os
import sys


def generate_html(image_paths, below, captions, output_file):
    """
    Generate an HTML file displaying images in rows with captions and text below each image.

    Parameters:
        image_paths (list of lists): Paths for images; image_paths[i][j] is the path for row i, column j.
        below (list of lists): Texts below images; below[i][j] is the text for row i, column j.
        captions (list): Captions for each row; captions[i] is the caption for row i.
        output_file (str): Output HTML file name.
    """
    # Start of HTML document
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Gallery</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 0;
            }
            .section {
                margin: 30px 0;
            }
            .caption {
                font-size: 1.5em;
                font-weight: bold;
                margin: 20px 0;
            }
            .image-row {
                display: flex;
                justify-content: center;
                gap: 15px;
                flex-wrap: wrap;
            }
            .image-container {
                text-align: center;
            }
            img {
                width: 150px;
                height: auto;
                border: 2px solid #ccc;
                border-radius: 8px;
            }
            .below-text {
                margin-top: 5px;
                font-size: 1em;
                color: #333;
            }
        </style>
    </head>
    <body>
    """

    # Add sections for each row of images
    for row_idx, row_images in enumerate(image_paths):
        caption = captions[row_idx] if row_idx < len(captions) else ""
        html_content += f'<div class="section">\n'
        html_content += f'    <div class="caption">{caption}</div>\n'
        html_content += f'    <div class="image-row">\n'

        for col_idx, image_path in enumerate(row_images):
            text_below = below[row_idx][col_idx] if row_idx < len(below) and col_idx < len(below[row_idx]) else ""
            html_content += f"""
                <div class="image-container">
                    <img src="{image_path}" alt="Image {row_idx + 1}-{col_idx + 1}">
                    <div class="below-text">{text_below}</div>
                </div>
            """
        html_content += f'    </div>\n</div>\n'

    # End of HTML document
    html_content += """
    </body>
    </html>
    """

    # Write to output file
    with open(output_file, "w") as f:
        f.write(html_content)
    print(f"HTML file successfully generated: {output_file}")
