# Real system dataset for "*Snapshot ptychography on array cameras*"

Welcome to the repository for the real system dataset of ["*Snapshot ptychography on array cameras*"](https://doi.org/10.1364/OE.447499). This dataset offers a collection of images and corresponding measurements obtained using array cameras that are purpose-built for *snapshot ptychography*.

## Data collection
The ground truth dataset is a collection obtained through a combination of downloading from [Openclipart](https://openclipart.org/) and generating new images using image augmentation techniques. To capture the ground truth data, they were displayed on a spatial light modulator. The measurement dataset were then recorded using array cameras, where a superluminescent light-emitting diode (LED) provided the illumination. Please refer to the ["*Snapshot ptychography on array cameras*"](https://doi.org/10.1364/OE.447499) for more detailed information about the data collection process.

## Data contents
The dataset is divided into:  
- Ground truth data: 27,000 binary images in directory "real_system_dataset_ground_truth/"
- Measurements data: 27,000 measurement images in directory "real_system_dataset_measurements/"  

The dataset is organized as follows:
```
-real_system_dataset_ground_truth/
  -00000.png
  -00001.png
  -...
  -26999.png
-real_system_dataset_measurements/
  -0.npy
  -1.npy
  -...
  -26999.npy
-README.md  
```
The corresponding measurements image of "00000.png" is "0.npy", and the subsequent files follow the same pattern.

### File formats
- Ground truth data
  - dimension: 600 $×$ 800 (H $\times$ W)
  - format: PNG
- Measurements data
  - dimension: 16 $×$ 576 $×$ 768 (C $\times$ H $\times$ W)
  - format: .npy

## Data usage
### Prediction
The neural network model takes measurement data as input. For more detailed information, please refer to this [notebook](https://github.com/djbradyAtOpticalSciencesArizona/arrayCameraFourierPtychography/blob/main/demo.ipynb) .

## Data Access
To access the dataset, please follow these steps:
1. Download this dataset from [UA ReDATA Repository](https://figshare.com/s/0dac12a23676c7150fc6).
2. Extract all zip files.
3. Navigate to the desired directory (real_system_dataset_ground_truth/ or real_system_dataset_measurements/) to access the corresponding files.

## Related materials
- Code: [Analysis code in GitHub](https://github.com/djbradyAtOpticalSciencesArizona/arrayCameraFourierPtychography)
- Paper: [*Wang, Chengyu, et al. "Snapshot Ptychography on Array cameras."*](https://doi.org/10.1364/OE.447499) (Optics Express)

## Maintainers
- [Chengyu Wang](https://github.com/ChengyuWang1007)
- [Xiao Wang](https://github.com/ShawnWong-wx)

## Copyright
This dataset is licensed under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contribution and Support
We welcome contributions and appreciate feedback to improve the dataset and its value for the research community. If you have any questions, suggestions, or would like to contribute, please contact us through djbrady@arizona.edu.
