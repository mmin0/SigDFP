# SigDFP
Signatured Deep Fictitious Play for Mean Field Games with Common Noises

<p align="center">
  <img src="algoGraph.png" width="600" height="450" />
</p>

## Installment
```
# need signature package
pip install -r requirements.txt
```
## Experiments
We have three experiments: SystemicRisk, Invest, InvestConsumption. To train model for examples, use the following code with corresponding case name and signature depth. For example, training SystemicRisk example with signature depth 2, run
```
python3 run.py --case SystemicRisk --depth 2
```


## Plots
Plots can be done by
```
python3 SystemicRiskPlot.py --depth 2
python3 InvestPlot.py --depth 2
python3 InvestConsumpPlot.py --depth 4
```
The depth should be the same as the depth for training.

## Citation

This code is for the paper "Signatured Deep Fictitious Play for Mean Field Games with Common Noise", if you find this useful in your research project, please cite
```
@InProceedings{pmlr-v139-min21a,
  title = 	 {Signatured Deep Fictitious Play for Mean Field Games with Common Noise},
  author =       {Min, Ming and Hu, Ruimeng},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {7736--7747},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/min21a/min21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/min21a.html},
  abstract = 	 {Existing deep learning methods for solving mean-field games (MFGs) with common noise fix the sampling common noise paths and then solve the corresponding MFGs. This leads to a nested loop structure with millions of simulations of common noise paths in order to produce accurate solutions, which results in prohibitive computational cost and limits the applications to a large extent. In this paper, based on the rough path theory, we propose a novel single-loop algorithm, named signatured deep fictitious play (Sig-DFP), by which we can work with the unfixed common noise setup to avoid the nested loop structure and reduce the computational complexity significantly. The proposed algorithm can accurately capture the effect of common uncertainty changes on mean-field equilibria without further training of neural networks, as previously needed in the existing machine learning algorithms. The efficiency is supported by three applications, including linear-quadratic MFGs, mean-field portfolio game, and mean-field game of optimal consumption and investment. Overall, we provide a new point of view from the rough path theory to solve MFGs with common noise with significantly improved efficiency and an extensive range of applications. In addition, we report the first deep learning work to deal with extended MFGs (a mean-field interaction via both the states and controls) with common noise.}
}
```
