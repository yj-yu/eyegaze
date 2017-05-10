Eye gaze + Pupil prediction
========

Setup
-----

### Basic Dependencies
Install dependencies from requirement list
```
pip install -r requirements.txt
```

### Python PATH

```
git submodule update --init --recursive
add2virtualenv .
```

### Install Required Packages
Install required packages:
* **Bazel** ([instructions](http://bazel.io/docs/install.html)).

### Prepare the Training Data

To train the model you will need to provide training data in native TFRecord format.
Please execute vrsumm/data/build_*_data.py files




Prepare Data
------------

* `dataset/VAS/`

To make symbolic link:

```
mkdir dataset
ln -sf /data1/common_datasets/VAS/ dataset/VAS/
```

See README.md for details.

Preprocess Scripts
------------------
We provide preprocess scripts in scripts folder
If you want more detail for preprocessing scripts,
See readme file in scripts

pip install git+https://github.com/wookayin/tensorflow-plot.git@master


Running Tests (Optional)
------------------------

