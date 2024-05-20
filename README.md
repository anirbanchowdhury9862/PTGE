# PTGE
Personal Transformer-Based Gaze Estimation
### Requirements
tensorflow==2.15.1
tf-models-official==2.15
### In case the notebooks are not rendered try them here https://nbviewer.org/
### Download MPIIGaze dataset http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz
### Download MPIIFaceGaze dataset http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip

### For preparing the data
```$python prep_data.py```
### Note: Edit the MPII dataset paths and the number of persons in prep_data.py
### Run the train_GazeModel.ipynb for implementation and training of the GazeModel
### Run the train_CalibrationModel.ipynb for implementation and training of the CalibrationModel and calibrate it for new user
### Run the test_calibration.ipynb for using newly estimated person embeddings from calibrationModel in the gazemodel

