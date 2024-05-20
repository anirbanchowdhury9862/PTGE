import tensorflow as tf
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

tf.keras.backend.clear_session()
effcnt_net=tf.keras.applications.EfficientNetV2B0(include_top=False,
                                            include_preprocessing=True,
                                            pooling=None)

effcnt_net.trainable=True
vgg16=tf.keras.applications.VGG16(include_top=False,pooling=None)
vgg16.trainable=True
vgg16_processor=tf.keras.applications.vgg16.preprocess_input

g_face=tf.keras.Model(inputs=effcnt_net.inputs,outputs=effcnt_net.outputs,name='g_face')
g_eye=tf.keras.Model(inputs=vgg16.inputs,outputs=vgg16.outputs,name='g_eye')

class GazeModel(tf.keras.Model):
    def __init__(self):
        super(GazeModel,self).__init__()
        self.g_face=g_face
        self.g_eye=g_eye
        self.flat=tf.keras.layers.Flatten()
        # Embedding layer as described in the paper
        self.embedding=tf.keras.layers.Embedding(3,6,
                                                 embeddings_regularizer=tf.keras.regularizers.L2(l2=0.01),
                                                 mask_zero=True,name='subject_embedding')
        #gradients wont pass through embedding layer upto 40 epochs
        self.embedding.trainable=False
        self.MLP=tf.keras.Sequential([
            tf.keras.layers.Dense(1280,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(3,name='gaze_location'),
            ],name='MLP')

    def call(self,input_dict):
        face_features=self.g_face(input_dict['face'])
        flipped_face_features=self.g_face(input_dict['flipped_face'])
        left_features=vgg16_processor(input_dict['lefteye'])
        left_features=self.g_eye(left_features)
        right_features=vgg16_processor(input_dict['righteye'])
        right_features=self.g_eye(right_features)
        face_features=self.flat(face_features)
        flipped_face_features=self.flat(flipped_face_features)
        left_features=self.flat(left_features)
        right_features=self.flat(right_features)
        embedding=self.embedding(input_dict['id'])
        rot_mat=input_dict['rotation_matrix']
        eye_coords=input_dict['eye_coords']
        total=tf.concat([face_features,flipped_face_features,left_features,
                            right_features,eye_coords,embedding,rot_mat],1)
        total=self.MLP(total)
        # return face_features, left_features
        return total
