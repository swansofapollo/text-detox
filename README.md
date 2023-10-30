## text-detoxification
 Text Detoxification Task is the process of converting toxic-styled text into neutral-styled text with the same meaning. I devised a method for detoxifying texts with a high level of toxicity.

## Data Description

The dataset is a subset of the ParaNMT corpus (50M sentence pairs). The filtered ParaNMT-detox corpus (500K sentence pairs) can be downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip). This is the main dataset for the assignment detoxification task.

The data is given in the `.tsv` format, means columns are separated by `\t` symbol.

| Column | Type | Discription | 
| ----- | ------- | ---------- |
| reference | str | First item from the pair | 
| ref_tox | float | toxicity level of reference text | 
| translation | str | Second item from the pair - paraphrazed version of the reference|
| trn_tox | float | toxicity level of translation text |
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |

## Dataset Availability
The dataset employed in the development of my text detoxification model is based on a comprehensive list of words deemed inappropriate by Google's standards. The list encompasses a broad spectrum of offensive and harmful language that my system is designed to detect and neutralize.

You can download the full list of bad words, which serves as a reference for my detoxification algorithm, from the following source:

Google's Bad Words List: https://www.freewebheaders.com/full-list-of-bad-words-banned-by-google/
This resource is pivotal for both training my models and for users interested in understanding the types of content targeted for detoxification.

## Model Download
The core of my detoxification system is encapsulated in a machine learning model that has been carefully trained and evaluated to ensure high efficacy in identifying and altering toxic language.

To utilize my solution, you may download the pre-trained model using the link provided below:

Detoxification Model: https://drive.google.com/file/d/1OQ8unkAmS18ijnkNKIUaSYgbj4Jc7MP-/view?usp=sharing
Alternatively, for those wishing to engage more deeply with the project or to reproduce the results from scratch, my Jupyter Notebook 2.1-final-solution.ipynb includes all necessary code and can be executed as described in the subsequent section.

##  Running the Code
To operationalize the text detoxification model, execute the Jupyter Notebook titled 2.1-final-solution.ipynb. It is essential to run the first and last three cells of the notebook to initialize the environment and conduct the final evaluations.

The initial cells prepare the environment by importing necessary libraries and setting up the machine learning pipeline, while the concluding cells are responsible for loading the model, processing the input data, and displaying the results.

The sequence for running the notebook is as follows:

- Start with the initial cells to establish your runtime environment and import dependencies.
- Conclude with the final three cells to apply the model to your dataset and obtain the detoxification results.
Ensure that you have a suitable Python environment with all required dependencies installed, including but not limited to TensorFlow, Transformers, and Hugging Face's datasets library.