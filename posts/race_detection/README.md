# Image Classification of X-Rays to Predict Ethnicity/Race

## Abstract

Deep learning image classification models have shown promise in identifying diseases through Chest X-rays. However, a series of results have also shown that with considerable accuracy, it is possible to determine the self-reported ethnicity of the patient. We aim to reproduce these results and outline the ethical concerns this poses both in terms of racism in medicine and the privacy of the patient.

## Motivation and Question

We want to use the publicly available ChexPert [here](https://stanfordmlgroup.github.io/competitions/chexpert/) database, which consists of 224,316 chest radiographs of 65,240 patients from Stanford Hospital, collected between October 2002 and July 2017. The data is mainly intended for classifying a variety of chest diseases, for which radiologists have provided separate annotations. This includes the self-reported age, gender, and ethnicity of the patient.

However, papers as "AI recognition of patient race in medical imaging: a modelling study" by Gichoya et. al [here](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext) have used the Chexpert dataset (along with other public and private chest radiographs) to accurately determine the self-reported age, gender, and ethnicity of the patient. That ethnicity can be detected is especially concerning as it may indicate that ML models for medical imaging classification can use race to make predictions. As they note, this could perpetuate existing inequalities in medical care for minority groups.

We want to see if it is possible to reproduce their results for ethnicity classification. The data they use, unsurprisingly, is a majority White, so we want to investigate whether our algorithm shows any bias in its classifications. Time permitting, it could also be worth investigating if this can be replicated for X-ray scans outside of the US.

We also want to investigate the ethical concerns with creating an algorithm that can identify the race of the patient. For one, we want to discuss its implications for medical imaging algorithms, which may use race in their predictions. Furthermore, a concern is that it may support racial determinism, the belief that race is a key factor in ones phenotypic abilities. To this end, we will discuss the paper's own investigations into the reasons why the algorithm can detect race as well as finding other papers on the impacts of racial determinism. Another concern we may investigate is de-anonymization. While we will not try to de-anonymize the data, it may be worthwhile discussing the impacts that having this information on age, gender, and ethnicity can have on revealing a patient's identity.

## Planned Deliverables

We plan to create a model that can predict a person's race based on the image of their chest X-ray, a Jupyter notebook that demonstrates the efficacy of this model, and a discussion of the ethical ambiguity of this model.

If everything works out as planned, we will have an image classification model that works to high accuracy and a beautiful analysis of its efficacy in a Jupyter notebook. We will also have a chance to analyze the factors that contribute to the model's accuracy; we can look at the most interesting features and the potential biases that may be present in the data. We will attempt to answer ethical questions posed by the success of this classification model: whether people of different races have different bone structures and what this difference may mean; whether this model can be used in a clinical setting, who it can benefit, who it can destroy, and how.

If our code is not successful, we will try to preexisting algorithms or code to conduct our analysis. Our deliverables will still contain a Python package containing our own code up to the point where things stop working. In the Jupyter notebook, we will use a preexisting package to show how close our model is to the 'standard' in terms of efficacy. We will also explain exactly why we fail: there may not be enough data; the rib cages of people from different ethnic backgrounds may not look very different (we may need to look for research papers on this); etc. The ethics research will look roughly the same as in the case that we actually succeed.

## Risk Statement

Given the previous research, we are confident that it is possible to create the algorithm. However, since we are all new to deep learning and image classification, there will likely be difficulties in training our algorithm and achieving the accuracies reported in the paper. Again, we do have references to help guide our approach in case we run into issues, but this is still a possible risk. It might be important to acknowledge that we don't have a clear goal of what features are important to look for, so when or if our model does classify ethnicities based off x-ray images, those features may not have any clear meaning to what actually differentiates the physical bodies of different races. The data is also a concern. So far, we have the ChexPert dataset, but it is 4-500 GB which would present a challenge. We are trying to access other datasets, such as the MIMIC-CXR, but this needs some credentialing which may take some time.

## Ethics Statement

We as a group do not support any system of racial discrimination. Potential ethical implications to this project stems from the racially charged motivations from phrenology as a medium to justify the hierarchies in races, slavery, and racist discourse. Thus, groups of people who have the potential to benefit from this project are medical professionals who can use this model to further analyze race and ethnicities as a factor in predicting for diseases; one consequence of this action is the violation of privacy and use of demographic data to assess the presence of medical conditions. On the other hand, groups of people who may be harmed are people of color whose vulnerabilities to medical processes and professionals can be abused and taken advantage of. Concluding this project can both have a negative and positive impact to the world. For instance, findings regarding a medical condition to certain ethnicities may further scientific literature to provide the necessary aid to prevent certain conditions for groups of ethnicities. However, the addition of race and ethnicity as a factor to the classification of x-ray images can be also used in other applications to negatively place racist boundaries on the basis of the body and bone structures.

## Resources Required

We need chest X-ray datasets that include the race of the patient. Right now we're considering using data from [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf). This has a .csv file with race labels for the patients in study 1. We're still looking for more data because knowledge is power.

As for algorithms, we may use `K-means` for image classification. `PyTorch` could be good, as they also have a X-ray learning package.

## Tentative Timeline

By week 9, we should have our data and should have visualizations of the demographics of the data and some explanations of the chest radiographs. It would be good if we have decided on a deep learning package to use by then and have set out a plan to train our algorithm.
