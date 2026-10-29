# Spoken script

Read this as one pass. It covers what was done on Roboflow, what each Python file does, and what the numbers mean.

## Opening

This project classifies a fabric photo into one of three labels: Defect free, Defected, or Waste. I built the same task twice. First I trained a ResNet-18 on Roboflow, with no training code of my own. Then I fine-tuned a ResNet-18 in PyTorch on the exact same split, so the comparison is about the platform and not about a bigger network. The report looks at three things: accuracy, training time, and inference latency, and then at the images both models get wrong.

## What I did on Roboflow, before any Python

I uploaded the fabric images and labeled each one with a single class. There are about five thousand source images. I generated dataset version 1 with three settings. The split is 70 percent train, 20 percent validation, and 10 percent test, which came out to 3,499 train, 999 validation, and 497 test. Every image was auto-oriented and stretched to 224 by 224, because that is the size ResNet-18 expects. Two augmentations were applied to the training images only: a horizontal flip half the time, and a brightness change of plus or minus 15 percent. Roboflow saved two copies of each training image, so the training folder has 6,998 files. Validation and test were not augmented. That matters, because the test score has to be on real images, not on flipped copies.

I then trained ResNet-18 from the ImageNet checkpoint and left the epochs, learning rate, and batch size at Roboflow's defaults. The model id is sanya-mohanty/poorva-fabric-defect-detection-idm-2-1-resnet18-t1. On the test set, Roboflow reports precision 94.5 percent, recall 91.7 percent, and F1 93.0 percent, at a confidence threshold of 61 percent.

The classes are uneven, and I say that before the accuracy numbers. After augmentation, training has 6,304 Defected images, 670 Defect free, and only 24 Waste. The test folder has 449 Defected, 47 Defect free, and 1 Waste. A model that mostly says Defected will look very accurate.

## How the code is arranged

Six Python files do the rest. common.py holds the model id and the shared list of 50 test images. download_dataset.py pulls version 1 from Roboflow. infer.py calls the hosted model and writes predictions.csv. train_pytorch.py fine-tunes ResNet-18 and times it. compare.py puts the two models side by side and copies the mistakes. report.py turns the saved numbers into the three-page PDF.

The private API key is read from the environment variable ROBOFLOW_API_KEY. It is never written into a file.

## download_dataset.py

This script logs in with the Roboflow SDK, opens the workspace sanya-mohanty, the project poorva-fabric-defect-detection-idm-2, and version 1. It downloads that version in folder format. Folder format means three directories, train, valid, and test, and inside each one a folder per class. The class folder name is the ground-truth label. That layout is what PyTorch's ImageFolder expects, and it is also how infer.py knows the true label of a test image: it is the name of the parent folder.

## common.py, the shared 50 images

The assignment asks for inference on 50 unseen images. Unseen means the test split, never train or validation. select_eval_images walks the test class folders and picks 50 images, taking turns across classes so one class cannot fill the list. The seed is 42. Waste has only one test image, so the list ends up as 25 Defect free, 24 Defected, and 1 Waste. The chosen paths are written to eval_images.txt. The next time any script runs, it reads that file instead of picking again. infer.py and train_pytorch.py therefore time and score the same 50 files.

## infer.py, the Roboflow API

infer.py builds an InferenceHTTPClient pointed at https://serverless.roboflow.com and the model id above. It loads the 50 images, then calls the model once on the first image and throws that time away. That warmup call is the slow first request, and it should not count as latency.

For each of the 50 images it records the clock, calls client.infer, and records the clock again. The difference is the latency. It includes uploading the image and waiting for the server. top_prediction reads the response and returns the winning class and its confidence. The hosted classifier puts those in the fields top and confidence.

predictions.csv has the three required columns: image name, predicted label, and confidence. A second file, results/roboflow/predictions_detailed.csv, adds the true label, whether the call was correct, and the latency. summary.json stores the score: 82 percent accuracy, mean latency 382 milliseconds, median 365, and a 95th percentile of 515 milliseconds.

## train_pytorch.py, the local ResNet-18

The script picks a device. On this Mac that is MPS, the Apple GPU. It would use CUDA on an NVIDIA machine, and the CPU if neither exists.

build_transforms makes two pipelines. Training resizes to 224, then randomly flips horizontally and jitters brightness by 15 percent. Those are the same two augmentations Roboflow used. Validation and test only resize, convert to a tensor, and normalize with ImageNet mean and standard deviation. The normalization has to match the ImageNet weights the model starts from.

ImageFolder reads each split. The folder names become the class list: Defect free, Defected, Waste. DataLoader serves them in batches of 32. Training batches are shuffled. Validation and test batches are not.

The model is torchvision's ResNet-18 with ImageNet weights. ResNet-18's last layer, called fc, was built for 1,000 ImageNet classes. I replace it with a new linear layer that has one output per fabric class, so three outputs. Those three numbers are the scores for the three labels.

Training uses cross-entropy loss and AdamW with learning rate 0.0003 and weight decay 0.0001. A cosine schedule lowers the learning rate across 15 epochs. Each epoch calls run_epoch twice. With an optimizer, it trains: forward pass, loss, backward pass, and an optimizer step. Without an optimizer, it only measures validation loss and accuracy, and gradients are off. After each epoch, if validation accuracy is the best so far, the weights are copied and kept. If validation does not improve for 4 epochs, training stops early. This run used all 15 epochs. The best validation accuracy was 97.0 percent at epoch 14. Wall-clock time was 19.2 minutes, about 77 seconds an epoch. The timer calls sync so the GPU finishes the work before the clock stops. Without that, the measured time would end before the GPU was done.

The best weights are saved as results/pytorch/best_resnet18.pt and then loaded back for testing. predict runs the full test loader, turns the three scores into probabilities with softmax, and takes the class with the highest probability. That produces the confusion matrix and the per-class precision, recall, and F1. Overall test accuracy is 95.8 percent, 476 of 497. Macro F1 is 91.4 percent. Defected recall is 98.2 percent. Defect free recall is 72.3 percent, 34 of 47. Waste is 1 of 1.

Latency is measured separately, on the same 50 images, one image at a time, after 10 warmup passes. Two clocks are kept. latency_ms includes opening the file and preprocessing. forward_ms is the network only. The means are 11.3 milliseconds and 8.8 milliseconds. All of this is written to results/pytorch/metrics.json.

## compare.py

compare.py loads the detailed Roboflow CSV and the PyTorch CSV and checks that they contain the same file names. It prints accuracy and mean latency for both. Then it keeps every image that either model got wrong, writes results/failure_cases/failure_cases.csv, and copies those images into that folder. There are 11 such images.

## What the numbers say

On the full test set the two models are close. Roboflow's F1 is 93.0 percent. PyTorch's macro F1 is 91.4 percent, with 95.8 percent accuracy. That accuracy is high largely because 449 of the 497 test images are Defected, and the model is good at that class.

On the shared 50 images, which are about half Defect free, Roboflow scores 82 percent and PyTorch scores 78 percent. Roboflow is ahead by two images.

Latency is the clear difference. The hosted call averages 382 milliseconds because the image goes over the network. The local model averages 11 milliseconds on the same laptop. PyTorch training took 19.2 minutes. Roboflow's training duration is in the completion email and is the one number to copy into the report table.

## The mistakes

All 11 failures in the set of 50 are Defect free images predicted as Defected. Nine are wrong in both models, usually with confidence above 0.97. Two are wrong only in PyTorch, and on those Roboflow's correct score is about 0.62, just above its 61 percent threshold.

The full PyTorch test set makes 21 errors. Thirteen are Defect free called Defected. Eight are Defected called Defect free. Waste is never confused with the other classes. The missed swatches look alike after the resize and the compression. Smooth Defected images have about the same pixel variation as a typical Defect free image. The Defect free images called Defected are the most textured ones, in the same range as a typical Defected image. Both models are splitting the cloth by how much the weave varies, and on these edge cases that does not match the label. A higher confidence threshold does not remove the mistakes, because many of them are scored above 0.99.

## Closing

Same architecture, same split, same two augmentations. Accuracy is close, and Roboflow is slightly ahead on the 50 shared images. The local model is much faster to run, and it trained in 19.2 minutes on a laptop. The useful next step is more Defect free and Waste images, and a review of the labels where those two classes look the same, rather than a larger network.
