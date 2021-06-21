# IE-Baseline

Forked from [zhengyima/kg-baseline-pytorch](https://github.com/zhengyima/kg-baseline-pytorch).

## Data download and preprocessing

Download [DuIE 2.0](https://dataset-bj.cdn.bcebos.com/qianyan/DuIE_2_0.zip) and extract json files to `{project_root}/data` folder.

Transform and preprocess the data using `trans.ipynb`.

## Visualizing training statistics

Open a new terminal, install tensorboard with pip (versions higher than this may fail to work):

```bash
pip install tensorboard<2.4
```

Replace the `lab` in the notebook url with `/proxy/port_num/`. **Do** add the final slash at the end.