# Detecting Greenwashing Project

This was a group porject in which we were tasked with training a ML model to detect greenwashing.

The 2 provided files are extracts from the final notebook - the dataset and some other details are left out in order to satisfy an NDA.

"model.py" is what was used in the project to set up the model we imported from HuggingFace.

"gibberishRemoval.py" is what we used to clean the dataset - this uses 2 main packages: Nostril and gibDetector - the latter being a bi-gram that analyses the likelihood of pairs of characters appearing adjacanet to one another.