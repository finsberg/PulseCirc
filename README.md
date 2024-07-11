# PulseCirc (Coupled 3D-0D Model of Left Ventricle with a lumped circulation model)

PulseCirc is a computational software package that integrates a detailed three-dimensional (3D) representation of the left ventricle with a simplified zero-dimensional (0D) lumped parameter model of the entire circulatory system. This hybrid model aims to simulate the physiological interactions between the heart and the circulatory system. This enables detailed analysis of LV function in common pathological scenarios, i.e., aortic stenosis, mitral regurgitations. PulseCirc is highly versatile, accepting any 3D model defined as a class with the necessary methods, making it adaptable to be integrated with various research pipelines. 

## Installation instructions

### Install with pip
Clone the package and instiall using pip within the package directory
```
python3 -m pip install .
```
or alternatively install it as edittable for customization. 
```
python3 -m pip -e install .
```
