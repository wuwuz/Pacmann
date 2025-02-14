# Private Nearest Neighbor Search with Sublinear Computation and Communication


[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/username/repo/blob/master/LICENSE)

We implement a sublinear query cost private vector ANN search algorithm based on [PianoPIR](https://eprint.iacr.org/2023/452.pdf) and Graph-based ANN.

## Table of Content

- [Dependency](#Dependency)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dependency

1. Install Go 1.22.
2. Install the dependency of NGT. Instruction can be found here (https://github.com/yahoojapan/NGT)
3. Run ``go mod tidy`` to download the go module dependencies.
4. Download the SIFT dataset (http://corpus-texmex.irisa.fr/). Run ``sh SIFT-download.sh``. It takes a while -- the dataset is around 100GB. *Warning: it could take hours to download and extrat the files. It's recommended to run it in the background. Make sure the disk space is enough (>230GB).*

## Usage

1. To run the private search algorithm: ``sh run-private-search.sh``. See the report in ``private-search-report.txt``. The parameters can be changed by modifying the script file.

*Optional:*

1. To run the NGT non-private ANN for quality comparison: ``sh run-ngt-search.sh``. See the report in ``ngt-report.txt``.
2. To run the cluster-based algorithm for quality comparison: ``sh run-cluster-search.sh``. See the report in ``cluster-report.txt``. (Requiring the FAISS package. To download it: ``pip install faiss-cpu``)
3. To test the latency of an optimized inner product baseline (as what we used in the paper):
- a. ``cd graphann``
- b. ``go test -v -run InnerProduct`` (you can go into ``graphann_test.go`` and see the parameters)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contacts

mingxunz@andrew.cmu.edu
