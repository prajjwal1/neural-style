echo "Creating directory named model"
mkdir model
echo "Entering into model"
cd model
wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
echo "Dataset has been downloaded successfully"
cd ..
