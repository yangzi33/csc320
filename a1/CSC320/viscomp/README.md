## viscomp library

Welcome to CSC320: Introduction to Visual Computing!

For some of the assignments in this course, you will use and develop `viscomp`,
a shared library for image processing in Python.

[Link to the README for Assignment 1](app/a1/README.md)

## Installation

Create a conda environment:
```
conda create -n csc320a1 python=3.9
conda activate csc320a1
pip install --upgrade pip
```

Install requirements:
```
pip install -r requirements.txt
```

Install `viscomp`:
```
python setup.py develop
```

## App Installation (Optional)

We also provide an optional interactive app that you can play around with.
If you want to use the interactive parts of the code and the assignment, you need additional requirements:

```
pip install -r requirements_app.txt
```

### Windows-specific installs

For Windows users, you may have to go through the following additional steps in order to install `glumpy` in  `requirements.txt`:

-   Microsoft Visual C++ 14.0 or greater is required. Get it with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
-   Download and place `freetype.dll` in the top level of this assignment. Instructions and download links can be found [here](https://github.com/rougier/freetype-py/blob/master/README.rst#window-users)

### OSX-specific installs

For OSX users, you may have to also specifically install freetype with:

```
conda install freetype
```

## Running assignments

To run the apps, run:

```
python app/a1/a1_app.py --image-path /path/to/a/cool/image.png
```

## Tour of the Library

`app` contains scripts and interactive apps that you can run for the assignments.

`viscomp/ops` contains useful helper functions.

`viscomp/algos` contains the meat of the assignments that you will be implementing.
