## Synthetic Data Generation with Invertible Neural Networks Satisfying Privacy Preservation Principles
* **Authors**: Malgorzata Schwab, Ashis Kumer Biswas
* **Affiliation**: Computer Science and Engineering, University of Colorado Denver
* **Contacts**: [https://ashiskb.info](https://ashiskb.info)

### Abstract
This work explores the applicability of Invertible Neural Networks to the data privacy aspect of Trustworthy AI.  We research the topic of reversibility in deep neural networks and leverage their remarkable data reconstruction capabilities to build a framework for intelligent synthetic data generation, which is a way to protect data privacy in the process of building and training machine learning models.  We apply previous findings and principles regarding variational autoencoders, deep generative maximum-likelihood training and invertibility in neural networks to propose a robust and configurable network architecture enabling bias-free synthetic data generation, accompanied by a software-as-a-service systems engineering blueprint.  This effort aims at enhancing data democratization practices in machine learning while preserving privacy of sensitive data and protecting proprietary intellectual property that it contains.

### Code repository listing
* `01_INN_MNIST_Synthetic_8.ipynb` -- contains the experiment pipeline to work with the proposed invertible neural network based autoencoder.
* `requirements.txt` -- necessary packages used to create the python virtual environment to be able to run the codes.
* All other files and directories are to be considered utilities required to run the notebook without issue.

### Acknowledgements
This material is based upon work supported by National Science Foundation under [Grant No. NSF 2329919](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2329919&HistoricalAwards=false).
