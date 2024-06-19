# Source Code & Sample Data of FaultInsight

We provide the source code & sample data of FaultInsight model in this directory.

## What's Included:

`requirements.txt` tracks the python library dependency of FaultInsight, run the following command to install the environment:

```
pip install -r requirements.txt
```

`main.py` is the main script file we used for evaluting on the oncall host failure dataset. The script contains the all logic of data preprocessing, model building, model training and result evaluation.

`dataset_config.yaml` is the dataset config file that contains a list of failure cases for main script file to conduct fault diagnosis on. Each failure case is described with case file name, failure start & end time, ground-truth root cause metrics.

We provide a sample case in the `data` folder, and you are able to add your cases to the dataset config file and evaluate FaultInsight on these cases.