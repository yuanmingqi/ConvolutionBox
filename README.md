<div align='center'>
    <img src= 'https://github.com/Mingqi-Yuan/ADMP/blob/master/example/pulseai_logo.png'>
</div>

<h1 align="center">
   ConvolutionBox
</h1>

---
<p align="center">
    <em>ConvolutionBox is is a convenient API for convolution visualization, whicn can help you design the architecture of the CNN.</em>
    <br>
        <a >
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"> 
    </a>
    <a>
        <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Passing">
    </a>
    <a href="https://github.com/pyecharts/pyecharts/pulls">
        <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat" alt="Contributions welcome">
    </a>
    <a href="https://pypi.org/project/pyecharts/">
        <img src="https://img.shields.io/badge/python-3.x-blue.svg" >
    </a>
</p>
<div align='center'>
    <img src= 'https://github.com/Mingqi-Yuan/ConvolutionBox/blob/master/example/index.gif'>
</div>

---
# Introduction

Different tasks, different data, of course, different models.In orader to establish the best CNN architecture, 
we can use the **NAS** to find it, but it takes much time and the result is instable.Most of the time, we just need 
to try different moudle like '**Inception**', '**Continuous convolution**' and so on.**Tensorboard** is best visualization API that I've 
ever used, but it always need a complete network and generate log, which is convenient for training but not for designing,
because sometimes we just need to know the capability of the module.So I build the **ConvolutionBox**, which can figuratively
described as a 'Convolution calculator', and it will help you rapidly get the behavior of moudle.Hope you will enjoy that!   

<div align='center'>
    <img src= 'https://github.com/Mingqi-Yuan/ConvolutionBox/blob/master/example/pooling.png'>
</div>

# Language & Frame

Language:

Python3 + HTML + JavaScript

Frame:

Flask + Echarts

---
# Usage

## Convolution

## Pooling

## Extractor

Coming soon!

---

# Get it now
Clone it：
```
$ git clone https://github.com/Mingqi-Yuan/ADMP.git
```
or  you can get the zip file，then make a  **Flask project** with **PyCharm**.
```
$  wget https://codeload.github.com/Mingqi-Yuan/ADMP/zip/master
```


# Packages required
The moudle below is required for the ADMP:

|Moudle|
|:--:|
| **TensorFlow** |
| **Flask**|
| **json** | 
| **NumPy**|
|**OpenCV**|

The python environment of **Anaconda3** is recommended.

---

# License
[MIT License](LICENSE)
