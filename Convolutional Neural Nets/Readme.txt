I have used three CNN layers with ‘relu’ activation function and did batch normalization for all the layers with a dropout propability of 25%. 
Finally, I have used flatten function and dense layer with ‘relu’ activation followed by a dense with a activation function ‘softmax’, both with batch normalization and a dropout probability of 20%.
I have used adam optimizer with a learning rate of 0.01, Also while doing the model fit I have introduced a batch_size constraint of 64 and trained the model for 7 epochs. The final accuracy achieved was 93.08%.

A)	The file size for mnist-fashion.tflite : 5182364 bytes
	The file size for mnist-fashion-quant.tflite : 1299208 bytes
	The difference is 3883156 bytes
B)	The accuracy of the baseline file is: 92.24%
	The accuracy of the quantized file is: 92.2
	The change is .04%
