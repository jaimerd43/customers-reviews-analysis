Text Analytics – A Comparative Analysis of Brand Reviews - Semi-supervised Learning (SSL)
==============================

The objective of this report is to outline the analysis conducted on customer reviews of two brands, using a dataset of 5,722 entries. Given the significant number of missing entries in the "emotions" column, a semi-supervised learning (SSL) approach is adopted to accurately label these missing observations, drawing on both the existing labels and the customer reviews. This method enables an analysis of emotional responses across both brands. The insights gained from this analysis provide the brands with a better understanding of their performance.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- Project information.
    ├── data
    │   └── data     <- Data from the reviews of all customers of two brands.  
    │
    ├── docs               <- A default Sphinx project.
    │
    ├── .py code           
    │   └── NLP.py <- Project code. 
    │
    ├── Notebooks
    │   └── 3              <- Complete project.
    │
    ├── reports            <- Detailed report with recomendations to the company. Following CRISP-DM methodolody.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable so src can be imported
    └──  src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module.
        │
        ├── data           <- Download the data.
           └── data
        


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
