pdbvangogh is a package intended to enable style transfer on macromolecular structures with artistic backgrounds

Sample output of pdbvangogh for a Pre-Q1 Riboswitch overlaid on an image of the Portland, OR skyline in the style of Van Gogh's Starry Night.

![](https://github.com/rsmichael/pdbvangogh/blob/main/pdbvangogh_2l1v_pdx.png)

The current version of this package uses the "classic" approach of [Gatys 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) with some tweaks to assure fidelity of the styled image to the molecular structure. Updates will add other style transfer methodologies.

# Usage 

# Installation:

## Clone the repo

```
git clone https://github.com/rsmichael/pdbvangogh.git
```

## Build the environment

```
cd pdbvangogh
conda env create --file env.yml
```
## Activate the environment and install the package

```
conda activate pdbvangogh
pip install .
```

# Usage

pdbvangogh has one main API function that takes in:
    - the pdb ID of a macromolecularstructure of interest
    - the path of a background image
    - the path of a style image
    - the prefix for saved outputs

From the main repo directory, you can execute:

```python
import os
from pdbvangogh.api import pdbvangogh

pdbvangogh(background_image = 'src/tests/in/pdx.png', 
        pdb_id = '2l1v',
        style_image = 'src/tests/in/starry_night.png',
        save_prefix = 'pdbvangogh_test',
        out_dir = os.getcwd(),
        background_size = 100,
        content_size = 30)
```

This test will make a low resolution image named pdbvangogh_test_overlayed_image.png. For a higher resolution image, change the background_size and content_size parameters, e.g. the following. As a note, runtimes will be very long on a CPU-only machine. 

```python
import os
from pdbvangogh.api import pdbvangogh

pdbvangogh(background_image = 'src/tests/in/pdx.png', 
        pdb_id = '2l1v',
        style_image = 'src/tests/in/starry_night.png',
        save_prefix = 'pdbvangogh_test',
        out_dir = os.getcwd(),
        background_size = 800,
        content_size = 500)
```


Documentation for other parameters for the pdbvangogh enabling more custom use cases is coming soon.



