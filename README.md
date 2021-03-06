# Persian-tc 'CODE' repo

All code and docker files are included in this repo. Model Weights, checkpoints and etc are stored in the google drive folder linked below :

https://drive.google.com/drive/folders/1BwPaqPqMPAm6wZMh6mAE7BmJp_UJvvue?usp=sharing

Replace the code files in this repo with the ones existed in the Gdrive and you're good to go.

Persian-tc is a Pythonic project for Persian Sentiment Classification.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies in the requirements.txt .

```bash
pip install -r requirements.txt
```

## Usage

## Python script

Run this command :

```python

python api.py

```
And send a POST request to "http://127.0.0.1:5000/glove" or "http://127.0.0.1:5000/w2v" with your persian sentence and the result is gonna be like this for "قیمت ها بالاست" :

```python

negative : 0.8784139156341553, neutral : 0.04363475367426872, positive : 0.0779513269662857

```
## Docker

After creating your DOCKERFILE and docker-compose.yml files (or using the provided), run below command :

```python

docker-compose up

```
then you can be able to send POST request to the api.

In the case of getting port errors or connection refusions try this :

```python
docker run -it -p 5000:5000 <name_of_your_docker_image>
```
or try running the image file using docker desktop and send your request to the offered port.

# Trained Models

## Codes for data cleaning, model training and plots of the models operations in both gloVe and w2v embeddings are in the jupyter notebook files. Enjoy playing with the params :)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
