To re(train):
`$ python3 retrain.py --image_dir ~/document-classifier/tf_files/documents/`

To run the "guess":
`$ python3 label_image.py --graph=./tf_files/retrained_graph.pb --labels=./tf_files/retrained_labels.txt --output_layer=final_result --image=rg12.jpg --input_layer=Placeholder`
 
 
 start flask server:
 `FLASK_APP=start.py flask run`

virtualenv:
`virtualenv --system-site-packages -p python3 ./venv`
`source ./venv/bin/activate`

tensorflow:
`https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl`
