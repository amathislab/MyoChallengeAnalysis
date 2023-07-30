# Policy_analysis_learning_process

# Notebook of the semester project
# Analysis of policies at different stages of the learning process
Bastien Le Lan

under the supervision of Alberto Chiappa

Mathis group

The notebook is in the file src/BastienLeLan_Analysis_of_learning_process_at_different_stages.ipynb

The notebook needs to be run from the src folder of the project, only one modification was made to the original code of the library is to rename all model model.zip inside their respective folder, to be able to load the curiculum weights efficiently.

A workaround this is to use the pickled folder i will join to be able to directly load the weights from the folder and in the same time gain a lot of time.
In this pickled folder there are the curiculum weights, but also the embeddings of the cebra projections, to permit to load them directly and not to have to recompute them each time.

The notebook is very long and heavy and might cause visual studio to crash, so I recommand only to run the part of interest using the outline, each main part can normally be run independently from the others.
The notebook is divided in 3 main sections, the analysis for the unobserved physics, the analysis of the perceptron role, and the curiculum learning.
