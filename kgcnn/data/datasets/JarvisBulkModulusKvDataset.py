from kgcnn.data.datasets.JarvisBenchDataset2021 import JarvisBenchDataset2021


class JarvisBulkModulusKvDataset(JarvisBenchDataset2021):
    """Store and process :obj:`MatProjectJdft2dDataset` from `MatBench <https://matbench.materialsproject.org/>`__
    database. Name within Matbench: 'matbench_jdft2d'.

    Matbench test dataset for predicting exfoliation energies from crystal structure
    (computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT database.
    For benchmarking w/ nested cross validation, the order of the dataset must be identical to the retrieved data;
    refer to the Automatminer/Matbench publication for more details.

        * Number of samples: 636
        * Task type: regression
        * Input type: structure

    """

    def __init__(self, reload=False, verbose: int = 10, data_main_dir: str = None):
        r"""Initialize 'bulk_modulus_kv' dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
            data_main_dir (str): Path to main data directory. Default is None (uses system default).
        """
        import os
        if data_main_dir is None:
            # Try common paths
            possible_paths = [
                os.path.join(os.path.expanduser("~"), "datasets"),  # ~/datasets
                "/home/datasets",  # /home/datasets
                os.path.join(os.getcwd(), "datasets"),  # ./datasets
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    data_main_dir = os.path.dirname(path) if os.path.basename(path) == "jarvis_dft_3d_bulk_modulus_kv" else path
                    break
            if data_main_dir is None:
                data_main_dir = os.path.join(os.path.expanduser("~"), "datasets")

        super(JarvisBulkModulusKvDataset, self).__init__(
            "bulk_modulus_kv", reload=reload, verbose=verbose, data_main_dir=data_main_dir
        )
        self.label_names = "bulk_modulus_kv "
        self.label_units = "Gpa"
