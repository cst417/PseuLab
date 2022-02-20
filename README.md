# PseuLab

--STILL BEING WORKED ON--

An implementation of Pseudo Labelling, a semi-supervised learning technique that
makes use of both labelled and unlabelled data. It trains a model of your choice on the labelled
data first. Then, it uses that model to make predictions on the unlabelled data and adds the new labelled
data into the training set, giving you extra data points to train with.



TODO:
- Add confidence thresholds before adding the newly predicted labels back into the training set.
- Improve code and add exception handling.
- Allow shuffling of dataset.
- Maybe more things
