import pymol
from pymol import cmd
import requests
import os


def download_cif(pdb_id, directory):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)

    if response.status_code == 200:
        # Save the CIF file
        with open(os.path.join(directory, f"{pdb_id}.cif"), "wb") as file:
            file.write(response.content)
        print(f"{pdb_id}.cif downloaded successfully.")
    else:
        print(f"Failed to download CIF file for {pdb_id}. Error code: {response.status_code}")


def visualize_cif_and_save_image(cif_file_path, image_output_path, width=800, height=600):
    """
    Visualizes a CIF file using PyMOL and saves the visualization as an image.
    Parameters:
    - cif_file_path: str, path to the CIF file.
    - image_output_path: str, path where the output image will be saved.
    - width: int, width of the output image.
    - height: int, height of the output image.
    """
    # Initialize PyMOL
    pymol.finish_launching(["pymol", "-c"])  # '-c' for command-line only (no GUI)

    # Load the CIF file
    cmd.load(cif_file_path, "my_structure")
    # Customize the view, representation, etc.
    cmd.hide("everything", "all")
    cmd.show("cartoon", "my_structure")  # Show as cartoon
    cmd.color("cyan", "my_structure")  # Color the structure cyan
    # Set the viewport size
    cmd.viewport(width, height)
    # Optionally, ray trace the image for higher quality
    cmd.ray(width, height)
    # Save the visualization to an image file
    cmd.png(image_output_path)
    # Quit PyMOL
    cmd.quit()
    print(f"Image saved to {image_output_path}")
