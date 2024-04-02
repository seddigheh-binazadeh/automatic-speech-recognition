# Speech Recognition
Automatic Speech Recognition (ASR) is the technology that enables a computer or machine to transcribe spoken language into text. It involves processing and interpreting audio input to produce written text, allowing for the conversion of spoken words into a digital format.
Some of the most important applications of ASR include: Voice typing, Virtual Assistants, Transcription Services, 
In recent years, there has been significant progress in the field of automatic speech recognition, driven by advancements in deep learning techniques. 
Recent advances in deep learning for speech research have been discussed by Deng et al. (2013) at Microsoft [1]. They focused on the application of deep learning techniques, particularly deep neural networks (DNNs), for acoustic modeling in speech recognition. Their work emphasized the use of DNNs to learn complex hierarchical representations of speech data, enabling more accurate recognition of phonemes and words.

Passricha, V et al. [2] discussed the use of deep learning techniques, such as deep neural networks (DNNs), convolutional neural networks (CNNs), and recurrent neural networks (RNNs) in bidirectional architecture, for speech recognition. They highlighted the application of DNNs for acoustic modeling, CNNs for feature extraction, and RNNs for modeling temporal dependencies in speech data.

The use of unsupervised and semi-supervised learning techniques, such as clustering and generative models, for speech recognition, reviewed in [3]. These techniques were applied to learn acoustic and linguistic representations from unlabeled speech data, ultimately improving the performance of automatic speech recognition systems.

The conformer model is a type of neural network architecture that has been used for speech recognition tasks [4]. It combines convolutional layers to capture local dependencies and self-attention mechanisms to capture global dependencies in the input sequence. The model also includes feed-forward layers and layer normalization to improve training stability. Overall, the conformer model has shown promising results in speech recognition tasks due to its ability to effectively capture both local and global dependencies in the input data.

William Chan et al., [5] introduced a sequence-to-sequence model based on the Transformer architecture for end-to-end speech recognition tasks. The model incorporates self-attention mechanisms to capture long-range dependencies in the input sequence, making it well-suited for speech recognition tasks.

In this work, we utilize the LJSpeech dataset to develop an automatic speech recognition model aimed at converting speech into text. The LJSpeech dataset consists of approximately 13,100 high-quality speech audio files, each accompanied by a corresponding text transcription.

# Preprocessing

In this project we use Mel-spectrogram representation instead of raw audio.  Next, each clip was transformed into its Mel-spectrogram representation. A spectrogram is a visual depiction of a signal’s frequency composition over time. The Mel scale provides a linear scale for the human auditory system, and is related to Hertz by the following formula, where m represents Mels and f represents Hertz:
 $m=2595 〖log〗_10  (1+f/700)$
The processed clips were converted into Mel spectrogram representations, which visually illustrate a signal's frequency makeup over time using the Mel scale, a linear scale for human hearing. The Mel spectrogram helps our models understand sound as humans do. This is achieved by passing the raw audio through filter banks to obtain the Mel spectrogram, resulting in a shape of 128 x input dimension for each sample, indicating 128 filter banks used and input dimension time steps per clip. Then we applied specAugment (A Simple Data Augmentation Method for Automatic Speech Recognition [6]) that is available in pytorch and consists of warping the features, masking blocks of frequency channels, and masking blocks of time steps. 
Figures 1 and 2 show a raw audio clip and its corresponding Mel frequency representation after augmentation. Our models learn features from this representation, and their architectures are discussed next.

![image](https://github.com/seddigheh-binazadeh/automatic-speech-recognition/blob/main/imgs/raw_audio_SpecAug.png)

# Model

Our model has two important parts. First, there is the feature extractor part, for which we utilize ResNet18 without the last 4 layers. We also add a MaxPool2D layer with a kernel size and stride of 2 for the end layer, and then permute and reshape the output size so that we obtain a feature vector of size [batch_size] x [seq_length] x [dimensions * channels].  Since the number of channels depends on the number of filters in the last convolutional layers, and the dimension depends on the size of the kernel and the stride of the convolutional layers and the freq_bins of the Mel spectrogram the product of dimension and channels is a static variable and is equal to 1024 in our design, and we refer these parameters as d _model. We set d_model in the Embedding Layers and also in the transformer layer to be equal to 1024 too .but the seq_length varies for each batch. 
To ensure precision in our model, we incorporate positional encoding as well as src_mask and trg_mask into the transformer model.
For Regularization we apply the dropout = 0.1 in transformer layer.
Our model has 127 M parameters. 

| Maxpool2d       | Embedding layer                    | Transformer Layer     |
| :-------------: | :---------------------------------:|:---------------------:|
| kernel_size = 2 | num_embedding = size of dictionary | num_encoder = 6       |
|  stride = 2     |  Dim_embedding = 1024 (d_model)    | num_decoder = 6       |
|        -        |                -                   | Attention heads = 4   |
|        -        |                -                   | d_model  = 1024       |
|        -        |                -                   | dim_FeedForward = 4096|

![image](https://github.com/seddigheh-binazadeh/automatic-speech-recognition/blob/main/imgs/model%20block.png)

# Experiment 
 we trained the model  with SGD optimizer by lr = (0.5 for 24 epoch and reduce lr to 0.05 for last 10 epoch), momentum = 0.9 .
Figure below represent the loss curves in training progress:

![image](https://github.com/seddigheh-binazadeh/automatic-speech-recognition/blob/main/imgs/loss_.png)

# Refrences 
1.	Deng, L., Li, J., Huang, J.T., Yao, K., Yu, D., Seide, F., Seltzer, M., Zweig, G., He, X., Williams, J. and Gong, Y., 2013, May. Recent advances in deep learning for speech research at Microsoft. In 2013 IEEE international conference on acoustics, speech and signal processing (pp. 8604-8608). IEEE.

2.	Passricha, V. and Aggarwal, R. (2020) A Hybrid of Deep CNN and Bidirectional LSTM for Automatic Speech Recognition. Journal of Intelligent Systems, Vol. 29 (Issue 1), pp. 1261-1274. https://doi.org/10.1515/jisys-2018-0372

3.	Taha, K. (2023). Semi-supervised and un-supervised clustering: A review and experimental evaluation. Information Systems, 102178.‏

4.	Gulati, A., Qin, J., Chiu, C. C., Parmar, N., Zhang, Y., Yu, J., ... & Pang, R. (2020). Conformer: Convolution-augmented transformer for speech recognition. arXiv preprint arXiv:2005.08100.‏

5.	Chan, W., Jaitly, N., Le, Q. V., & Vinyals, O. (2015). Listen, attend and spell. arXiv preprint  arXiv:1508.01211.‏

6.	Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). Specaugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779.‏



