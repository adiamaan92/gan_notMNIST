TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.gan_keras"
PACKAGE_STAGING_PATH="gs://gan-nonmnist/notMNIST/stage"
JOB_DIR="gs://gan-nonmnist/notMNIST/tmp"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="MNIST_$now"
REGION="us-central1"
DATA="gs://gan-nonmnist/notMNIST.pickle"

gcloud ml-engine jobs submit training $JOB_NAME \
	--job-dir $JOB_DIR \
	--package-path $TRAINER_PACKAGE_PATH \
	--module-name $MAIN_TRAINER_MODULE \
	--region $REGION




	


