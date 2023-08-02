# cvmlp_recruiting_f23

Create the conda environment:
```bash
conda_env_name=cvmlp
conda create -n $conda_env_name python=3.9 -y &&
conda activate $conda_env_name &&

# Mamba is used for much, much faster installation.
conda install mamba -y -c conda-forge &&
# Remove cuda if you aren't using a GPU
mamba install \
  habitat-sim=0.2.4 headless pytorch==1.13.1 pytorch-cuda=11.6 \
  -c aihabitat -c pytorch \
  -c nvidia -c conda-forge -y


# Installing habitat-lab (make sure your env is active): 
git clone --branch v0.2.4 git@github.com:facebookresearch/habitat-lab.git &&
cd habitat-lab &&
pip install -e habitat-lab &&
pip install -e habitat-baselines &&
cd ..

```