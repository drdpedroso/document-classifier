To re(train):
`$ python3 retrain.py --image_dir ~/document-classifier/tf_files/documents/`

To run the "guess":
`$ python3 label_image.py --graph=./tf_files/retrained_graph.pb --labels=./tf_files/retrained_labels.txt --output_layer=final_result --image=rg12.jpg --input_layer=Placeholder`
 