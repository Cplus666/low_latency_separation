# An Effective MVDR Post-Processing Method For Low-Latency Convolutive Blind Source Separation
This repository implements a low-latency blind source separation system using the **Minimum Variance Distortionless Response (MVDR)** architecture. The system also incorporates **Weighted Prediction Error (WPE)** to reduce reverberation and improve the separation performance by **Auxiliary Function Independent Vector Analysis (AuxIVA)**.

## Usage
To run the blind source separation algorithm:

```
python main_offline_implementation.py
```

The output files will be saved in "save_wav" directory

## Citation
If you find this work useful, please cite it in your research:

```
@inproceedings{chua2024an,
  title={An Effective MVDR Post-Processing Method For Low-Latency Convolutive Blind Source Separation},
  author={Chua, Jiawen and Yan, Longfei Felix and Kleijn, W Bastiaan},
  booktitle={2024 IEEE International Workshop on Acoustic Signal Enhancement (IWAENC)},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgements
This implementation builds on the **Weighted Prediction Error (WPE)** and **Auxiliary Function Independent Vector Analysis (AuxIVA)** methods for blind source separation.

### WPE
The WPE method used for reverberation reduction is based on the following work:
Drude, Lukas, et al. "Nara-WPE: A python package for weighted prediction error dereverberation in Numpy and Tensorflow for online and offline processing." ICASSP 2018.

Github Repository: https://github.com/fgnt/nara_wpe

### AuxIVA
The source separation is also inspired by the AuxIVA method:
Ono, Nobutaka. "Stable and fast update rules for independent vector analysis based on auxiliary function technique." IEEE WASPAA 2011.

Github Repository: https://github.com/onolab-tmu/libss

Please cite these works if you use this project in your research.