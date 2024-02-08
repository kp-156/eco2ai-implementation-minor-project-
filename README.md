# eco2ai-implementation-minor-project

### Setup Guide

Create a new anaconda environment using the following command:
```
conda create -n eco2ai python=3.10
```

Activate the environment using the following command:
```
conda activate eco2ai
```

Install eco2ai package using pip as follows:
1. Clone the eco2ai repository: https://github.com/sb-ai-lab/Eco2AI
2. Navigate to the root directory of the repository
3. Run the following command:
```
pip install -e .
```

Browse to the root directory of this repository and run the following command to install other dependencies as follows:
```
pip install -r requirements.txt
```

### Running the code

Browse to the root directory of this repository and run the following command to run the code:
```
python main.py
```

To run different models, add a model run script in the models folder, and change the model import in the main.py file.
To run variations of the same model, change the parameters in the main.py file.
