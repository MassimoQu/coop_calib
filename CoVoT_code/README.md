
## Model Brief
CoVoT is a novel collaborative perception framework designed to address the trade-off between communication efficiency and perception performance in current vehicular systems. Through sparse feature processing and a dynamic attention window voting mechanism, CoVoT achieves efficient cross-vehicle collaborative perception with ultra-low communication overhead. As the first fully sparse feature processing-based collaborative perception framework, CoVoT significantly enhances perception accuracy while dramatically reducing communication costs.


## Data Downloading
All the data can be downloaded from [google drive](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu). If you have a good internet, you can directly
download the complete large zip file such as `train.zip`. In case you suffer from downloading large fiels, we also split each data set into small chunks, which can be found 
in the directory ending with `_chunks`, such as `train_chunks`. After downloading, please run the following command to each set to merge those chunks together:
```python
cat train.zip.parta* > train.zip
unzip train.zip
```

## Installation
Our model is built upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), please refer to [data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html) and [installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare data and install OpenCOOD. To see more details of OPV2V data, please check [website of opv2v.](https://mobility-lab.seas.ucla.edu/opv2v/)

We also offer our  `requirement.txt` file for pip installation.

## Quick Start

### Train your model
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/second_early_fusion.yaml`, meaning you want to train an early fusion model which utilizes SECOND as the backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.

We release our model at https://huggingface.co/Anonymous-CoVoT/CoVoT.

### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v_data_dumping/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.

We release our model at https://huggingface.co/Anonymous-CoVoT/CoVoT.

### Data sequence visualization
To quickly visualize the LiDAR stream in the OPV2V dataset, first modify the `validate_dir` in your `opencood/hypes_yaml/visualization.yaml` to the opv2v data path on your local machine, e.g. `opv2v/validate`, and the run the following commond:
```python
cd ~/OpenCOOD
python opencood/visualization/vis_data_sequence.py [--color_mode ${COLOR_RENDERING_MODE}]
```
Arguments Explanation:
- `color_mode` : str type, indicating the lidar color rendering mode. You can choose from 'constant', 'intensity' or 'z-value'.








