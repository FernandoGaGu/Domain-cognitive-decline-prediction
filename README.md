# <mark>TBD. Add final title</mark>

Repository with the code associated with research article: <mark>TBD. Add title and link</mark>.

## Events

⚠️ This repository is still a work in progress

The addition of code and documentation is still ongoing. We've tried to keep things as organized as possible, but, as often happens in research, the process took many unexpected turns, making it hard to leave behind perfectly clean and well-documented code.

We're doing our best to improve things, and we really appreciate your patience while we continue polishing everything! 🙏


- (2025-06-02) Repository creation 🚀
- (2025-06-02) Added codes for:
    - Processing ADNI clinical data.
    - Detecting anomalies in neuroimaging data.
    - Computing connectivity matrices.
    - Base code of the models

    
## Disclaimer

The neuroimaging, clinical, and neuropsychological data used in this project were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (https://adni.loni.usc.edu/). As such, this repository does not host, distribute, or manage access to any of the original data.

All researchers seeking to use ADNI data must apply for access directly through the official ADNI Data Sharing and Publications Policy, and comply with all associated terms and conditions. For more information about data access, usage restrictions, and citation requirements, please visit:

🔗 https://adni.loni.usc.edu/data-samples/access-data/

Please note:

- This repository only contains code and scripts developed for data processing and analysis.

- Any data preprocessing steps shown here assume the user has legitimate access to ADNI data.

- Users are responsible for ensuring their own compliance with ADNI's data usage agreements.

Moreover, some of the data processing steps (mainly for neuroimaging processing) described in this repository rely on third-party software tools that are proprietary and subject to licensing restrictions.


## Repository structure

### `notebook`

Directory with different notebooks used to process information and perform analysis:

- **ADNI-Preprocessing-clinical-data-preprocessing.ipynb**. Notebook used to create the clinical dataset from ADNI neuropsychological assessments and clinical evaluations.
- **ADNI-Composites-calculation.ipynb**. Notebook used for the calculation of the composite scores.
- **ADNI-Final-database-mount.ipynb**. Notebook used to build the final databases used for modeling. This notebook depends on *ADNI-Preprocessing-clinical-data-preprocessing.ipynb* and *ADNI-Composites-calculation.ipynb*.
- **Neuroimaging-Outlier-detection.ipynb**. Notebook used to detect artifacts in the processed images.
- **Neuroimaging-Graphical-lasso.ipynb**. Notebook used to calculate the brain connectivity matrix based on FDG-PET baseline data used as input for the graph neural networks. 
- **Splits-generation.ipynb**. Code used to generate the different splits used for training / validation and testing.



### `src`

Directory with different scripts and libraries used in the study.


