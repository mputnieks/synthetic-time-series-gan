# synthetic-time-series-gan
 
Quickstart guide:
> extract data from https://github.com/irinagain/Awesome-CGM/wiki/Colas-(2019) to path data/online
> set up venv, pytorch and CUDA (see useful_scripts.txt for guidance)
> run compiler-online.py
> encode file using processing-pipeline-healthgan.py
(see commented code from compiler-dialect.py for guidance)
> place the resulting sdv file in data/train and train the GAN