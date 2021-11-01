# Array Camera Ptychography

This repository provides the code for the following papers:

[*Schulz, Timothy J., David J. Brady, and Chengyu Wang. "Photon-limited bounds for phase retrieval."*](https://doi.org/10.1364/OE.425796) (Optics Express)

*Wang, Chengyu, Minghao Hu, Yuzuru Takashima, Timothy J. Schulz, and David J. Brady. "Snapshot Ptychography on Array cameras."*

### Photon-limited bounds for phase retrieval
The optimal Cramér-Rao lower bound on the mean-square error for the estimation of a coherent signal from photon-limited intensity measurements is equal to the number of signal elements, or the number of signal elements minus one when we account for the unobservable reference phase. Whereas this bound is attained by phase-quadrature holography, we also show that it can be attained through a phase-retrieval system that does not require a coherent reference. 

***Fourier_Ptychography_for_Phase_Retrieval.ipynb*** : We implemented a Fourier ptychographic system where sampling windows were applied to the signal in the Fourier space, and the selected patches were inverse transformed to generate low-resolution frames. 

### Snapshot Ptychography on Array-cameras

**Physucal setup**:
![](https://github.com/djbradyAtOpticalSciencesArizona/arrayCameraFourierPtychography/blob/main/proto_system.png)

**Demonstration of the groundtruth and the corresponding measurements**:
![](https://github.com/djbradyAtOpticalSciencesArizona/arrayCameraFourierPtychography/blob/main/sample_measurement.gif)

***demo.ipynb***: a demo of the reconstruction results.

Please contact the author for the access to the entire dataset.

### Citation
If you find the code useful in your research, please consider citing:

    @article{schulz2021photon,
       author = {Schulz, Timothy J and Brady, David J and Wang, Chengyu},
        title = {Photon-limited bounds for phase retrieval},
      journal = {Optics Express},
         year = {2021},
       number = {11},
       volume = {29},
        pages = {16736--16748},
    publisher = {Optical Society of America}
    }
    

